"""Tests for src/universe.py -- constituent CSV parsing/fetch-with-fallback,
universe sync, and the checkpointed/resumable bulk OHLCV fetch pipeline.

This is the network-facing layer of the project. Every test here mocks
`requests.get`, `yfinance`, and the shared `_fetch_validate_upsert_one` core
-- nothing in this file makes a real network call or touches the real
project database (data/trading_data.duckdb). Where a scenario would
otherwise depend on src/validation/validate_ohlcv.py's exact check
thresholds, we patch `validate_ohlcv` itself to a controlled stub instead --
that module has its own test suite elsewhere, and coupling to its internal
rules here would make these tests fragile to changes that aren't universe.py's
concern.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src import universe as universe_module
from src.universe import (
    _fetch_index_constituents,
    _fetch_validate_upsert_one,
    _filter_error_severity_rows,
    _has_sufficient_coverage,
    _to_storage_symbols,
    _write_checkpoint,
    bulk_fetch_and_store,
    ensure_fetch_checkpoint_schema,
    ensure_universe_schema,
    fetch_nifty50_constituents,
    fetch_nifty500_constituents,
    get_active_universe,
    resumable_bulk_fetch,
    sync_universe,
)
from tests.helpers import insert_ohlcv, make_conn, seed_active_universe


# --------------------------------------------------------------------------
# _parse_constituent_csv (via _fetch_index_constituents, which is the only
# public entry point that reaches it end to end)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


def test_fetch_index_constituents_parses_first_successful_url(monkeypatch):
    """The first URL that returns a valid CSV wins -- would catch a bug that
    always falls through to the manual CSV or a later URL even on success."""
    csv_text = "Symbol,Company Name\nRELIANCE,Reliance Industries Ltd\nTCS,Tata Consultancy Services Ltd\n"
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(csv_text)

    monkeypatch.setattr(universe_module.requests, "get", fake_get)

    records = _fetch_index_constituents(("https://a.example/x.csv", "https://b.example/y.csv"), None, "TestIndex")

    assert calls == ["https://a.example/x.csv"]  # never tried the second URL
    assert records == [
        {"symbol": "RELIANCE", "company_name": "Reliance Industries Ltd", "yf_ticker": "RELIANCE.NS"},
        {"symbol": "TCS", "company_name": "Tata Consultancy Services Ltd", "yf_ticker": "TCS.NS"},
    ]


def test_fetch_index_constituents_falls_back_to_second_url(monkeypatch):
    """First URL fails, second succeeds -- would catch a bug that gives up
    after the first failure instead of trying the rest of csv_urls."""
    csv_text = "Symbol,Company Name\nINFY,Infosys Ltd\n"

    def fake_get(url, headers=None, timeout=None):
        if url == "https://a.example/x.csv":
            raise RuntimeError("connection refused")
        return _FakeResponse(csv_text)

    monkeypatch.setattr(universe_module.requests, "get", fake_get)

    records = _fetch_index_constituents(("https://a.example/x.csv", "https://b.example/y.csv"), None, "TestIndex")
    assert records == [{"symbol": "INFY", "company_name": "Infosys Ltd", "yf_ticker": "INFY.NS"}]


def test_fetch_index_constituents_falls_back_to_manual_csv(monkeypatch, tmp_path):
    """Every live URL fails -> falls back to the manual CSV rather than
    raising immediately. Would catch a bug that skips the manual fallback."""

    def fake_get(url, headers=None, timeout=None):
        raise RuntimeError("all live sources down")

    monkeypatch.setattr(universe_module.requests, "get", fake_get)

    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text("symbol,company_name\nHDFC,HDFC Bank Ltd\n")

    records = _fetch_index_constituents(("https://a.example/x.csv",), manual_csv, "TestIndex")
    assert records == [{"symbol": "HDFC", "company_name": "HDFC Bank Ltd", "yf_ticker": "HDFC.NS"}]


def test_fetch_index_constituents_raises_when_everything_fails(monkeypatch, tmp_path):
    """Live URLs fail AND the manual CSV is missing -> RuntimeError, not a
    silent empty list. Would catch a bug that swallows the failure."""
    monkeypatch.setattr(universe_module.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    missing_csv = tmp_path / "does_not_exist.csv"
    with pytest.raises(RuntimeError):
        _fetch_index_constituents(("https://a.example/x.csv",), missing_csv, "TestIndex")


def test_parse_constituent_csv_boundary_cases():
    """Exercises _parse_constituent_csv's own edge cases directly through
    the fetch-with-fallback path (it has no other public entry)."""
    import io

    from src.universe import _parse_constituent_csv

    # blank/NaN symbol rows are dropped, symbols are upper-cased and stripped
    records = _parse_constituent_csv("Symbol,Company Name\n reliance ,Reliance Ltd\n,Missing Symbol Co\nTCS,TCS Ltd\n")
    assert records == [
        {"symbol": "RELIANCE", "company_name": "Reliance Ltd"},
        {"symbol": "TCS", "company_name": "TCS Ltd"},
    ]

    # missing company-name column -> company_name defaults to ""
    records = _parse_constituent_csv("Symbol\nWIPRO\n")
    assert records == [{"symbol": "WIPRO", "company_name": ""}]

    # missing symbol column entirely -> explicit ValueError
    with pytest.raises(ValueError, match="symbol column"):
        _parse_constituent_csv("Ticker,Company Name\nWIPRO,Wipro Ltd\n")

    # header only, no data rows -> explicit ValueError, not an empty list
    with pytest.raises(ValueError, match="No constituents"):
        _parse_constituent_csv("Symbol,Company Name\n")


def test_fetch_nifty50_and_nifty500_constituents_use_their_own_urls_and_labels(monkeypatch):
    """fetch_nifty50/500_constituents are thin wrappers -- verify each passes
    ITS OWN url tuple/manual path/label through, not the other index's (would
    catch the two wrappers' constants being swapped)."""
    captured = {}

    def fake_fetch(csv_urls, manual_csv_path, index_label):
        captured["csv_urls"] = csv_urls
        captured["manual_csv_path"] = manual_csv_path
        captured["index_label"] = index_label
        return [{"symbol": "X", "company_name": "X Ltd", "yf_ticker": "X.NS"}]

    monkeypatch.setattr(universe_module, "_fetch_index_constituents", fake_fetch)

    fetch_nifty50_constituents()
    assert captured["csv_urls"] == universe_module.NIFTY50_CSV_URLS
    assert captured["manual_csv_path"] == universe_module.MANUAL_NIFTY50_CSV
    assert captured["index_label"] == "Nifty 50"

    fetch_nifty500_constituents()
    assert captured["csv_urls"] == universe_module.NIFTY500_CSV_URLS
    assert captured["manual_csv_path"] == universe_module.MANUAL_NIFTY500_CSV
    assert captured["index_label"] == "Nifty 500"


# --------------------------------------------------------------------------
# sync_universe / get_active_universe
# --------------------------------------------------------------------------


def test_sync_universe_adds_removes_and_never_duplicates_active_rows():
    """Two syncs: (A,B,C) then (B,C,D). Would catch a bug that re-inserts a
    still-active symbol as a duplicate row, or fails to mark a dropped
    symbol inactive."""
    conn = make_conn()
    ensure_universe_schema(conn)

    def constituents(symbols):
        return [{"symbol": s, "yf_ticker": f"{s}.NS"} for s in symbols]

    summary1 = sync_universe(conn, constituents(["A", "B", "C"]), index_name="NIFTY50")
    assert summary1 == {"added": 3, "removed": 0, "active": 3}

    summary2 = sync_universe(conn, constituents(["B", "C", "D"]), index_name="NIFTY50")
    assert summary2 == {"added": 1, "removed": 1, "active": 3}

    # exactly one row per symbol ever inserted -- B and C were NOT re-inserted
    row_count = conn.execute("SELECT COUNT(*) FROM universe").fetchone()[0]
    assert row_count == 4  # A, B, C, D -- A's original row persists, just marked inactive

    a_row = conn.execute(
        "SELECT is_active, removed_date FROM universe WHERE symbol = 'A'"
    ).fetchone()
    assert a_row[0] is False
    assert a_row[1] == date.today()

    active = get_active_universe(conn, index_name="NIFTY50")
    assert active == ["B.NS", "C.NS", "D.NS"]


def test_get_active_universe_as_of_date_boundaries():
    """Directly seeds rows with fixed added_date/removed_date to pin down
    get_active_universe's inclusive/exclusive boundary exactly, independent
    of sync_universe's non-injectable date.today(). Would catch an off-by-one
    on either the added_date (<=) or removed_date (>) comparison."""
    conn = make_conn()
    ensure_universe_schema(conn)
    conn.execute(
        """
        INSERT INTO universe (symbol, yf_ticker, index_name, added_date, removed_date, is_active)
        VALUES ('AAA', 'AAA.NS', 'NIFTY50', '2024-01-01', '2024-06-01', false)
        """
    )

    # before added_date: not yet active
    assert get_active_universe(conn, "NIFTY50", "2023-12-31") == []
    # exactly on added_date: active (added_date <= as_of is inclusive)
    assert get_active_universe(conn, "NIFTY50", "2024-01-01") == ["AAA.NS"]
    # the day before removal: still active
    assert get_active_universe(conn, "NIFTY50", "2024-05-31") == ["AAA.NS"]
    # exactly on removed_date: NOT active (removed_date > as_of is exclusive)
    assert get_active_universe(conn, "NIFTY50", "2024-06-01") == []
    # after removal: not active
    assert get_active_universe(conn, "NIFTY50", "2024-06-02") == []


# --------------------------------------------------------------------------
# _filter_error_severity_rows
# --------------------------------------------------------------------------


def _candles(dates):
    return pd.DataFrame({"timestamp": pd.to_datetime(dates), "close": range(len(dates))})


def _issues(rows):
    """rows: list of (date_str, severity)."""
    return pd.DataFrame(
        [
            {"symbol": "X", "date": d, "issue_type": "test_issue", "severity": sev, "details": ""}
            for d, sev in rows
        ]
    )


def test_filter_error_severity_rows_drops_only_error_severity_dates():
    """A warning-severity row must survive; an error-severity row's candle
    must be dropped. Would catch a severity check that's inverted or that
    drops warnings too (silently losing good data)."""
    candles = _candles(["2024-01-01", "2024-01-02", "2024-01-03"])
    issues = _issues([("2024-01-01", "warning"), ("2024-01-02", "error")])

    kept = _filter_error_severity_rows(candles, issues)
    assert sorted(pd.to_datetime(kept["timestamp"]).dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-03"]


def test_filter_error_severity_rows_noop_on_empty_issues_or_candles():
    candles = _candles(["2024-01-01"])
    empty_issues = _issues([])
    assert len(_filter_error_severity_rows(candles, empty_issues)) == 1
    assert _filter_error_severity_rows(pd.DataFrame(columns=["timestamp"]), _issues([("2024-01-01", "error")])).empty


# --------------------------------------------------------------------------
# _fetch_validate_upsert_one
# --------------------------------------------------------------------------


def _fake_yfinance_df(dates, closes):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.DatetimeIndex(pd.to_datetime(dates), name="Date"),
    )


def test_fetch_validate_upsert_one_success_stores_rows(monkeypatch):
    """Happy path: mocked yfinance data with no validation issues lands in
    ohlcv_data via the real upsert_ohlcv, and the outcome reports the actual
    fetched range. Would catch a broken wiring between fetch/prepare/upsert."""
    conn = make_conn()
    raw_df = _fake_yfinance_df(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])

    monkeypatch.setattr(universe_module, "_fetch_yfinance_history", lambda **kw: raw_df)
    monkeypatch.setattr(universe_module, "validate_ohlcv", lambda candles, symbol: pd.DataFrame())

    outcome = _fetch_validate_upsert_one(conn, "RELIANCE.NS", "2024-01-01", "2024-01-04", "1d")

    assert outcome == {
        "success": True,
        "rows": 3,
        "first_date": "2024-01-01",
        "last_date": "2024-01-03",
        "error": None,
    }
    stored = conn.execute("SELECT COUNT(*) FROM ohlcv_data WHERE symbol = 'RELIANCE'").fetchone()[0]
    assert stored == 3


def test_fetch_validate_upsert_one_empty_yfinance_result_is_a_reported_failure(monkeypatch):
    """yfinance returning an empty DataFrame (e.g. delisted/wrong ticker) is
    a failure outcome, not a crash and not a silent empty success."""
    conn = make_conn()
    monkeypatch.setattr(universe_module, "_fetch_yfinance_history", lambda **kw: pd.DataFrame())

    outcome = _fetch_validate_upsert_one(conn, "FAKE.NS", "2024-01-01", "2024-01-04", "1d")
    assert outcome["success"] is False
    assert "no rows" in outcome["error"]


def test_fetch_validate_upsert_one_filters_error_rows_but_keeps_the_rest(monkeypatch):
    """One row flagged error-severity, others clean -- the error row must be
    dropped from storage while the rest still get stored and reported.
    Would catch error-severity filtering not actually being applied before upsert."""
    conn = make_conn()
    raw_df = _fake_yfinance_df(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 101.0, 102.0])

    monkeypatch.setattr(universe_module, "_fetch_yfinance_history", lambda **kw: raw_df)
    monkeypatch.setattr(
        universe_module,
        "validate_ohlcv",
        lambda candles, symbol: _issues([("2024-01-02", "error")]),
    )
    monkeypatch.setattr(universe_module, "log_issues", lambda conn, issues_df: None)

    outcome = _fetch_validate_upsert_one(conn, "RELIANCE.NS", "2024-01-01", "2024-01-04", "1d")
    assert outcome["success"] is True
    assert outcome["rows"] == 2
    stored_dates = sorted(
        d.strftime("%Y-%m-%d")
        for d in conn.execute("SELECT timestamp FROM ohlcv_data").df()["timestamp"]
    )
    assert stored_dates == ["2024-01-01", "2024-01-03"]


def test_fetch_validate_upsert_one_all_rows_error_severity_is_a_failure(monkeypatch):
    """Every row flagged error-severity -> nothing survives to store, and
    that's reported as a failure (not a false 'success' with 0 rows)."""
    conn = make_conn()
    raw_df = _fake_yfinance_df(["2024-01-01", "2024-01-02"], [100.0, 101.0])

    monkeypatch.setattr(universe_module, "_fetch_yfinance_history", lambda **kw: raw_df)
    monkeypatch.setattr(
        universe_module,
        "validate_ohlcv",
        lambda candles, symbol: _issues([("2024-01-01", "error"), ("2024-01-02", "error")]),
    )
    monkeypatch.setattr(universe_module, "log_issues", lambda conn, issues_df: None)

    outcome = _fetch_validate_upsert_one(conn, "RELIANCE.NS", "2024-01-01", "2024-01-04", "1d")
    assert outcome["success"] is False
    assert "failed validation" in outcome["error"]


# --------------------------------------------------------------------------
# bulk_fetch_and_store
# --------------------------------------------------------------------------


def test_bulk_fetch_and_store_partitions_successes_and_failures(monkeypatch):
    """Would catch a bug that mixes up which tickers land in 'successful' vs
    'failed', or that aborts the whole batch on one failure."""
    conn = make_conn()
    outcomes = {
        "A.NS": {"success": True, "rows": 1, "first_date": "2024-01-01", "last_date": "2024-01-01", "error": None},
        "B.NS": {"success": False, "rows": 0, "first_date": None, "last_date": None, "error": "boom"},
        "C.NS": {"success": True, "rows": 1, "first_date": "2024-01-01", "last_date": "2024-01-01", "error": None},
    }
    monkeypatch.setattr(
        universe_module, "_fetch_validate_upsert_one",
        lambda conn, yf_ticker, start_date, end_date, interval: outcomes[yf_ticker],
    )

    result = bulk_fetch_and_store(conn, ["A.NS", "B.NS", "C.NS"], "2024-01-01", "2024-01-04", delay_seconds=0)
    assert result == {"successful": ["A.NS", "C.NS"], "failed": ["B.NS"]}


def test_bulk_fetch_and_store_never_sleeps_after_the_last_ticker(monkeypatch):
    """Boundary: len(tickers)-1 sleeps for N tickers, never a trailing sleep
    after the last one. Regresses an off-by-one that would add a pointless
    delay (or one too few) at the end of a batch."""
    conn = make_conn()
    sleep_calls = []
    monkeypatch.setattr(universe_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(
        universe_module, "_fetch_validate_upsert_one",
        lambda conn, yf_ticker, start_date, end_date, interval: {
            "success": True, "rows": 1, "first_date": "2024-01-01", "last_date": "2024-01-01", "error": None,
        },
    )

    bulk_fetch_and_store(conn, ["A.NS"], "2024-01-01", "2024-01-04", delay_seconds=5)
    assert sleep_calls == []  # single ticker -> zero sleeps

    sleep_calls.clear()
    bulk_fetch_and_store(conn, ["A.NS", "B.NS", "C.NS"], "2024-01-01", "2024-01-04", delay_seconds=5)
    assert sleep_calls == [5, 5]  # three tickers -> exactly two sleeps


# --------------------------------------------------------------------------
# checkpoint schema / _write_checkpoint / _has_sufficient_coverage
# --------------------------------------------------------------------------


def test_write_checkpoint_upserts_by_symbol():
    """Re-writing a checkpoint for the same symbol updates the existing row
    rather than duplicating it -- the whole point of a resumable job's status
    journal being one-row-per-symbol."""
    conn = make_conn()
    ensure_fetch_checkpoint_schema(conn)

    _write_checkpoint(conn, "AAA", "AAA.NS", "pending", None, None, None, None)
    _write_checkpoint(conn, "AAA", "AAA.NS", "success", 10, "2024-01-01", "2024-01-10", None)

    rows = conn.execute("SELECT status, rows_fetched FROM fetch_checkpoint WHERE symbol = 'AAA'").fetchall()
    assert rows == [("success", 10)]


@pytest.mark.parametrize(
    "existing_min,existing_max,expected",
    [
        ("2024-01-11", "2024-01-20", True),   # min exactly at start+tolerance(10d) -> covered
        ("2024-01-12", "2024-01-20", False),  # min one day past tolerance -> not covered
        ("2024-01-01", "2024-01-10", True),   # max exactly at end-tolerance(10d) -> covered
        ("2024-01-01", "2024-01-09", False),  # max one day short of tolerance -> not covered
    ],
)
def test_has_sufficient_coverage_tolerance_boundary(existing_min, existing_max, expected):
    """Pins the exact <=/>= tolerance boundary from the docstring: start_date
    2024-01-01, end_date 2024-01-20, tolerance_days=10. Would catch an
    off-by-one (< vs <=) that wrongly re-fetches (or wrongly skips) data
    sitting exactly at the tolerance edge."""
    conn = make_conn()
    insert_ohlcv(conn, "AAA", [(existing_min, 100, 100), (existing_max, 100, 100)])

    result = _has_sufficient_coverage(conn, "AAA", "2024-01-01", "2024-01-20", "1d", tolerance_days=10)
    assert result is expected


def test_has_sufficient_coverage_no_existing_data_is_false():
    conn = make_conn()
    assert _has_sufficient_coverage(conn, "NEWCO", "2024-01-01", "2024-01-20", "1d", tolerance_days=10) is False


# --------------------------------------------------------------------------
# resumable_bulk_fetch
# --------------------------------------------------------------------------


def test_resumable_bulk_fetch_skips_already_covered_and_classifies_full_vs_partial(monkeypatch):
    """The core 'resumable' contract: a symbol already sufficiently covered
    in ohlcv_data is skipped and never reaches the fetch function; the rest
    are fetched and classified full/partial by how close first_date landed
    to the requested start. Would catch: a resumed run re-fetching data it
    already has, or full/partial classification being swapped."""
    conn = make_conn()
    # AAA already has full coverage for the requested range -> must be skipped.
    insert_ohlcv(conn, "AAA", [("2024-01-01", 100, 100), ("2024-01-20", 100, 100)])

    fetch_calls = []

    def fake_fetch_one(conn, yf_ticker, start_date, end_date, interval):
        fetch_calls.append(yf_ticker)
        if yf_ticker == "BBB.NS":
            return {"success": True, "rows": 5, "first_date": "2024-01-01", "last_date": "2024-01-05", "error": None}
        if yf_ticker == "CCC.NS":
            # first_date well after start+tolerance -> partial history (e.g. recent listing)
            return {"success": True, "rows": 3, "first_date": "2024-01-15", "last_date": "2024-01-18", "error": None}
        return {"success": False, "rows": 0, "first_date": None, "last_date": None, "error": "boom"}

    monkeypatch.setattr(universe_module, "_fetch_validate_upsert_one", fake_fetch_one)

    result = resumable_bulk_fetch(
        conn, ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"],
        "2024-01-01", "2024-01-20", delay_seconds=0, coverage_tolerance_days=10,
    )

    assert fetch_calls == ["BBB.NS", "CCC.NS", "DDD.NS"]  # AAA never reached the fetch function
    assert result["skipped"] == ["AAA.NS"]
    assert result["succeeded_full"] == ["BBB.NS"]
    assert result["succeeded_partial"] == ["CCC.NS"]
    assert result["failed"] == [("DDD.NS", "boom")]
    assert result["rows_added"] == 8  # 5 (BBB) + 3 (CCC); AAA contributed 0 since it was skipped

    checkpoint_status = dict(
        conn.execute("SELECT symbol, status FROM fetch_checkpoint").fetchall()
    )
    assert checkpoint_status == {"AAA": "skipped", "BBB": "success", "CCC": "success", "DDD": "failed"}


# --------------------------------------------------------------------------
# _to_storage_symbols
# --------------------------------------------------------------------------


def test_to_storage_symbols_strips_ns_suffix_and_leaves_others_alone():
    assert _to_storage_symbols(["RELIANCE.NS", "TCS.NS", "^NSEI"]) == ["RELIANCE", "TCS", "^NSEI"]
    assert _to_storage_symbols([]) == []

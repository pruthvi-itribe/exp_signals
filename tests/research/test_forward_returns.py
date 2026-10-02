"""Correctness tests for research/forward_returns.py.

compute_forward_returns is tested as a pure function on small hand-picked
panels; store_forward_returns's upsert semantics are tested against a bare
in-memory DuckDB connection (it creates its own `forward_returns` schema).
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from research.forward_returns import (
    FORWARD_RETURN_COLUMNS,
    _resolve_symbols,
    compute_and_store_forward_returns,
    compute_forward_returns,
    ensure_forward_returns_schema,
    store_forward_returns,
)


def _panel(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


# ---------------------------------------------------------------------------
# compute_forward_returns
# ---------------------------------------------------------------------------


def test_compute_forward_returns_hand_computed_values():
    """AAA compounds +10%/day for 5 days -> fwd_return_1d is 0.10 at every
    row except the last (no future row); fwd_return_2d is 0.21 (1.1^2 - 1)
    except the last two rows. Would catch a wrong shift direction (backward
    instead of forward), an off-by-one horizon, or the tail rows being
    dropped instead of left as NaN.
    """
    df = _panel("AAA", [100.0, 110.0, 121.0, 133.1, 146.41])
    result = compute_forward_returns(df, horizons=[1, 2])

    assert len(result) == 5  # tail rows kept, not dropped
    assert result["fwd_return_1d"].iloc[:4].tolist() == pytest.approx([0.1, 0.1, 0.1, 0.1], rel=1e-9)
    assert pd.isna(result["fwd_return_1d"].iloc[4])

    assert result["fwd_return_2d"].iloc[:3].tolist() == pytest.approx([0.21, 0.21, 0.21], rel=1e-9)
    assert pd.isna(result["fwd_return_2d"].iloc[3])
    assert pd.isna(result["fwd_return_2d"].iloc[4])


def test_compute_forward_returns_symbol_isolation():
    """Two symbols with very different price paths, concatenated. Would
    catch a bug where a forward-shift operation crosses the symbol boundary
    (e.g. a raw .shift(-h) applied to the whole concatenated frame instead
    of per-group) and computes AAA's last-row forward return using BBB's
    first price.
    """
    aaa = _panel("AAA", [100.0, 200.0, 100.0])  # last row's true fwd_return_1d is NaN
    bbb = _panel("BBB", [9999.0, 9999.0, 9999.0])  # deliberately huge, easy to spot a leak
    df = pd.concat([aaa, bbb], ignore_index=True)

    result = compute_forward_returns(df, horizons=[1])
    aaa_out = result[result["symbol"] == "AAA"].reset_index(drop=True)
    bbb_out = result[result["symbol"] == "BBB"].reset_index(drop=True)

    assert aaa_out["fwd_return_1d"].iloc[0] == pytest.approx(1.0)  # 200/100 - 1
    assert aaa_out["fwd_return_1d"].iloc[1] == pytest.approx(-0.5)  # 100/200 - 1
    assert pd.isna(aaa_out["fwd_return_1d"].iloc[2])  # NOT computed from BBB's 9999

    assert bbb_out["fwd_return_1d"].iloc[0] == pytest.approx(0.0)
    assert bbb_out["fwd_return_1d"].iloc[1] == pytest.approx(0.0)
    assert pd.isna(bbb_out["fwd_return_1d"].iloc[2])


def test_compute_forward_returns_empty_input():
    """Would catch a groupby-on-empty-frame crash."""
    df = pd.DataFrame(columns=["symbol", "date", "adj_close"])
    result = compute_forward_returns(df, horizons=[1, 5])
    assert list(result.columns) == ["symbol", "date", "fwd_return_1d", "fwd_return_5d"]
    assert result.empty


# ---------------------------------------------------------------------------
# _resolve_symbols
# ---------------------------------------------------------------------------


def test_resolve_symbols_strips_ns_suffix_when_given_explicitly():
    """Would catch the .NS-stripping normalization being dropped or applied inconsistently."""
    conn = duckdb.connect(":memory:")
    result = _resolve_symbols(conn, ["RELIANCE.NS", "TCS", "INFY.NS"])
    assert result == ["RELIANCE", "TCS", "INFY"]
    conn.close()


def test_resolve_symbols_falls_back_to_active_universe(monkeypatch):
    """When symbols=None, falls back to get_active_universe(). Would catch
    the fallback being skipped (e.g. always returning an empty list instead
    of calling through)."""
    import research.forward_returns as forward_returns_module

    monkeypatch.setattr(
        forward_returns_module, "get_active_universe", lambda conn: ["FOO.NS", "BAR.NS"]
    )
    conn = duckdb.connect(":memory:")
    result = _resolve_symbols(conn, None)
    assert result == ["FOO", "BAR"]
    conn.close()


# ---------------------------------------------------------------------------
# store_forward_returns
# ---------------------------------------------------------------------------


def test_store_forward_returns_empty_df_is_a_no_op():
    """Would catch an empty DataFrame being upserted as a zero-value row instead of skipped entirely."""
    conn = duckdb.connect(":memory:")
    summary = store_forward_returns(conn, pd.DataFrame())
    assert summary == {"rows_inserted": 0, "rows_updated": 0, "total_rows": 0}
    conn.close()


def test_store_forward_returns_rejects_unsupported_horizon_column():
    """forward_returns has room for exactly FORWARD_RETURN_COLUMNS -- would
    catch a caller passing an arbitrary extra horizon silently getting
    dropped instead of raising."""
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "fwd_return_99d": [0.1]})
    with pytest.raises(ValueError, match="unsupported columns"):
        store_forward_returns(conn, df)
    conn.close()


def test_store_forward_returns_upsert_replaces_not_duplicates():
    """Re-storing a row at the same (symbol, date) key must UPDATE the
    existing row, not insert a duplicate -- the core upsert contract. Would
    catch a plain INSERT with no ON CONFLICT clause (duplicate rows) or a
    conflict clause that fails to actually overwrite the value.
    """
    conn = duckdb.connect(":memory:")
    ensure_forward_returns_schema(conn)

    first = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "fwd_return_1d": [0.05]})
    summary1 = store_forward_returns(conn, first)
    assert summary1 == {"rows_inserted": 1, "rows_updated": 0, "total_rows": 1}

    second = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "fwd_return_1d": [0.09]})
    summary2 = store_forward_returns(conn, second)
    assert summary2 == {"rows_inserted": 0, "rows_updated": 1, "total_rows": 1}

    rows = conn.execute("SELECT fwd_return_1d FROM forward_returns").df()
    assert len(rows) == 1
    assert rows["fwd_return_1d"].iloc[0] == pytest.approx(0.09)
    conn.close()


def test_store_forward_returns_missing_columns_stored_as_null():
    """df may omit some of the four fwd_return_*d columns -- would catch
    those ending up as 0.0 (a very different, wrong value for "not yet
    known") instead of NULL."""
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-01"], "fwd_return_1d": [0.05]})
    store_forward_returns(conn, df)

    row = conn.execute(
        "SELECT fwd_return_1d, fwd_return_5d, fwd_return_10d, fwd_return_20d FROM forward_returns"
    ).df().iloc[0]
    assert row["fwd_return_1d"] == pytest.approx(0.05)
    assert pd.isna(row["fwd_return_5d"])
    assert pd.isna(row["fwd_return_10d"])
    assert pd.isna(row["fwd_return_20d"])
    conn.close()


# ---------------------------------------------------------------------------
# compute_and_store_forward_returns (integration)
# ---------------------------------------------------------------------------


def test_compute_and_store_forward_returns_no_ohlcv_returns_zero_summary(monkeypatch):
    """No OHLCV rows for the requested symbols -- would catch a crash on an
    empty frame instead of the documented all-zero summary."""
    from tests.helpers import make_conn

    conn = make_conn()
    summary = compute_and_store_forward_returns(conn, symbols=["NOPE"])
    assert summary == {"symbols": 0, "ohlcv_rows": 0, "rows_inserted": 0, "rows_updated": 0, "total_rows": 0}
    conn.close()


def test_compute_and_store_forward_returns_end_to_end():
    """Seeds real ohlcv_data rows, runs the full compute+store pipeline, and
    checks the stored fwd_return_1d against the same hand-computed value
    used in test_compute_forward_returns_hand_computed_values. Would catch a
    wiring bug between _load_daily_adj_close, compute_forward_returns, and
    store_forward_returns (e.g. wrong column passed through) that a test of
    each function in isolation could miss.
    """
    from tests.helpers import insert_ohlcv, make_conn

    conn = make_conn()
    insert_ohlcv(
        conn,
        "AAA",
        [
            ("2024-01-01", 100.0, 100.0),
            ("2024-01-02", 110.0, 110.0),
            ("2024-01-03", 121.0, 121.0),
        ],
    )
    summary = compute_and_store_forward_returns(conn, symbols=["AAA"])
    assert summary["symbols"] == 1
    assert summary["ohlcv_rows"] == 3
    assert summary["rows_inserted"] == 3

    row = conn.execute(
        "SELECT fwd_return_1d FROM forward_returns WHERE symbol = 'AAA' AND date = '2024-01-01'"
    ).df().iloc[0]
    assert row["fwd_return_1d"] == pytest.approx(0.1, rel=1e-9)
    conn.close()

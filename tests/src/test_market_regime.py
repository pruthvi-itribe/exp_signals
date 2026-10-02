"""Tests for src/market_regime.py: the Nifty 50 index-level regime filter
that bridges the "Nifty 50 market-regime filter" gap documented in
strategies/README.md and ARCHITECTURE.md.

Would catch: a wrong EMA-warm-up boundary, a "breakdown" that fires on every
bearish day instead of just the crossing day, regime values leaking across
symbols instead of being broadcast purely by date, or attach_market_regime
failing *closed* (silently blocking every entry) instead of *open* when
regime data is missing/incomplete -- the documented, deliberate contract.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from src.market_regime import (
    NIFTY_INDEX_SYMBOL,
    attach_market_regime,
    fetch_nifty_index_history,
    load_market_regime,
)
from tests.helpers import insert_ohlcv, make_conn


def test_load_market_regime_no_index_data_returns_empty_with_columns():
    """No ^NSEI history at all -- callers must treat this as "unavailable",
    not crash and not report an empty result as "always bearish"."""
    conn = make_conn()
    result = load_market_regime(conn)
    assert result.empty
    assert list(result.columns) == [
        "date", "nifty_close", "nifty_ema_100", "nifty_regime_bullish", "nifty_regime_breakdown",
    ]


def test_load_market_regime_ema_warmup_and_crossing_boundary():
    """100 constant days (EMA(100) exactly equals that constant throughout,
    by construction of an EWM of a constant series), then a jump up (still
    bullish), then a sharp drop that crosses below the EMA -- the single
    day that must be flagged as a breakdown -- then a second bearish day
    that must NOT re-flag (one-time crossing event, not persistent state).

    Would catch: an off-by-one in the min_periods warm-up window, treating
    every bearish day as a "breakdown" instead of just the crossing day, or
    an inclusive/exclusive mistake on the equal-to-EMA boundary (a close
    exactly on the EMA line must count as bullish, not bearish).
    """
    conn = make_conn()
    dates = pd.date_range("2020-01-01", periods=103, freq="D")
    closes = [100.0] * 100 + [110.0, 90.0, 90.0]
    rows = [(d.strftime("%Y-%m-%d"), c, c) for d, c in zip(dates, closes)]
    insert_ohlcv(conn, NIFTY_INDEX_SYMBOL, rows)

    regime = load_market_regime(conn)
    assert len(regime) == 103

    # Day 100 (index 99): first day EMA is defined; constant series -> ema == 100.0
    # exactly; close == ema -> counts as bullish (boundary is >=, not >).
    row99 = regime.iloc[99]
    assert row99["nifty_ema_100"] == pytest.approx(100.0)
    assert row99["nifty_regime_bullish"] == True  # noqa: E712
    assert row99["nifty_regime_breakdown"] == False  # noqa: E712

    # Day 101 (index 100): jumps to 110, still bullish (close > ema).
    row100 = regime.iloc[100]
    assert row100["nifty_regime_bullish"] == True  # noqa: E712
    assert row100["nifty_regime_breakdown"] == False  # noqa: E712

    # Day 102 (index 101): drops to 90, crosses below the EMA -- the breakdown day.
    row101 = regime.iloc[101]
    assert row101["nifty_close"] == pytest.approx(90.0)
    assert row101["nifty_close"] < row101["nifty_ema_100"]
    assert row101["nifty_regime_bullish"] == False  # noqa: E712
    assert row101["nifty_regime_breakdown"] == True  # noqa: E712

    # Day 103 (index 102): still bearish, but NOT a new breakdown (already
    # bearish yesterday) -- a one-time crossing event, not a persistent flag.
    row102 = regime.iloc[102]
    assert row102["nifty_regime_bullish"] == False  # noqa: E712
    assert row102["nifty_regime_breakdown"] == False  # noqa: E712


def test_load_market_regime_warmup_rows_fail_open_bullish():
    """Before the EMA is defined (fewer than 100 rows so far), regime state
    must default to bullish/no-breakdown, not NaN or bearish -- a strategy
    reading this before the index itself has enough history must behave as
    if the filter weren't gating anything yet."""
    conn = make_conn()
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows = [(d.strftime("%Y-%m-%d"), 100.0, 100.0) for d in dates]
    insert_ohlcv(conn, NIFTY_INDEX_SYMBOL, rows)

    regime = load_market_regime(conn)
    assert regime["nifty_ema_100"].isna().all()
    assert (regime["nifty_regime_bullish"] == True).all()  # noqa: E712
    assert (regime["nifty_regime_breakdown"] == False).all()  # noqa: E712


def test_attach_market_regime_broadcasts_by_date_not_by_symbol():
    """Two different symbols on the same date must get the identical regime
    values -- a market-wide condition, never a per-symbol one."""
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-03-01", "2024-03-01"]),
        }
    )
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01"]),
            "nifty_close": [100.0],
            "nifty_ema_100": [90.0],
            "nifty_regime_bullish": [True],
            "nifty_regime_breakdown": [False],
        }
    )
    result = attach_market_regime(df, regime_df)
    assert result["nifty_regime_bullish"].tolist() == [True, True]
    assert result["nifty_regime_breakdown"].tolist() == [False, False]


def test_attach_market_regime_fails_open_on_empty_regime_df():
    """No index data fetched at all -- must default to bullish/no-breakdown
    (the pre-feature behavior), never silently block every entry."""
    df = pd.DataFrame({"symbol": ["AAA"], "date": pd.to_datetime(["2024-03-01"])})
    result = attach_market_regime(df, pd.DataFrame())
    assert result["nifty_regime_bullish"].tolist() == [True]
    assert result["nifty_regime_breakdown"].tolist() == [False]


def test_attach_market_regime_fails_open_on_uncovered_dates():
    """A date outside regime_df's coverage (e.g. before the index was first
    fetched) must fail open too, not propagate NaN into a strategy's boolean masks."""
    df = pd.DataFrame(
        {"symbol": ["AAA", "AAA"], "date": pd.to_datetime(["2024-01-01", "2024-03-01"])}
    )
    regime_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-01"]),
            "nifty_close": [100.0],
            "nifty_ema_100": [110.0],
            "nifty_regime_bullish": [False],
            "nifty_regime_breakdown": [True],
        }
    )
    result = attach_market_regime(df, regime_df).set_index("date")
    # 2024-01-01 isn't in regime_df -> fails open.
    assert result.loc[pd.Timestamp("2024-01-01"), "nifty_regime_bullish"] == True  # noqa: E712
    # 2024-03-01 IS covered -> the real (bearish/breakdown) values pass through.
    assert result.loc[pd.Timestamp("2024-03-01"), "nifty_regime_bullish"] == False  # noqa: E712
    assert result.loc[pd.Timestamp("2024-03-01"), "nifty_regime_breakdown"] == True  # noqa: E712


def test_attach_market_regime_empty_df_returns_unchanged():
    empty = pd.DataFrame()
    assert attach_market_regime(empty, pd.DataFrame()) is empty


def test_fetch_nifty_index_history_delegates_to_bulk_fetch_and_store(monkeypatch):
    """Wiring test only (no real network call): fetch_nifty_index_history
    must fetch exactly the index symbol, under NIFTY_INDEX_SYMBOL, via the
    same shared fetch/validate/upsert pipeline every other symbol uses."""
    calls = []

    def fake_bulk_fetch_and_store(conn, tickers, start_date, end_date, **kwargs):
        calls.append((tickers, start_date, end_date))
        return {"successful": tickers, "failed": []}

    monkeypatch.setattr("src.market_regime.bulk_fetch_and_store", fake_bulk_fetch_and_store)

    conn = make_conn()
    result = fetch_nifty_index_history(conn, "2013-01-01", "2024-01-01")

    assert calls == [(["^NSEI"], "2013-01-01", "2024-01-01")]
    assert result == {"successful": ["^NSEI"], "failed": []}

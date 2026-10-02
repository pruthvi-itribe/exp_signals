"""Tests for src/trading_calendar.py: NSE trading-day lookups and gap
detection/logging against ohlcv_data.

Would catch: weekends/holidays leaking into "expected trading days", a
tz-naive/tz-aware mismatch shifting a calendar date by the UTC offset (this
project hit exactly that class of bug before), or
days_since_previous_available_date being computed from the wrong anchor date
when a symbol has more than one separate gap in range.
"""

import pandas as pd
import pytest

from src.trading_calendar import (
    find_missing_trading_days,
    get_nse_trading_days,
    log_gaps,
    mark_resolved,
)
from tests.helpers import insert_ohlcv, make_conn


def test_get_nse_trading_days_excludes_weekends():
    days = get_nse_trading_days("2024-01-01", "2024-01-19")
    assert all(d.weekday() < 5 for d in days)
    # Verified against the installed pandas_market_calendars 'NSE' calendar:
    # all 15 weekdays in this range are trading days (no holiday falls here).
    assert len(days) == 15


def test_get_nse_trading_days_tz_localize_none_keeps_the_same_calendar_date():
    """The calendar returns tz-aware midnight timestamps. Stripping tz via
    tz_localize(None) (what find_missing_trading_days does) must keep the
    same wall-clock date -- tz_convert(None) would instead shift by the UTC
    offset and silently move the date, which is the bug class this project
    hit before with tz-naive/aware comparisons."""
    days = get_nse_trading_days("2024-01-01", "2024-01-02")
    naive = days.tz_localize(None)
    assert list(naive.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]


def test_find_missing_trading_days_detects_single_gap():
    conn = make_conn()
    symbol = "GAPCO"
    for d in ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]:  # 2024-01-03 (Wed) missing
        insert_ohlcv(conn, symbol, [(d, 100, 101)])

    gaps = find_missing_trading_days(conn, symbol, "2024-01-01", "2024-01-05")

    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row["symbol"] == symbol
    assert pd.Timestamp(row["missing_date"]) == pd.Timestamp("2024-01-03")
    assert row["days_since_previous_available_date"] == 1  # 2024-01-02 -> 2024-01-03
    conn.close()


def test_find_missing_trading_days_no_gaps_returns_empty():
    conn = make_conn()
    symbol = "FULLCO"
    for d in ["2024-01-01", "2024-01-02", "2024-01-03"]:
        insert_ohlcv(conn, symbol, [(d, 100, 101)])
    gaps = find_missing_trading_days(conn, symbol, "2024-01-01", "2024-01-03")
    assert gaps.empty
    conn.close()


def test_find_missing_trading_days_all_weekend_range_has_no_expected_days():
    """2024-01-06/07 is a Sat/Sun -- zero expected trading days in range, so
    a symbol with no data at all there must still report zero gaps: nothing
    was missed because nothing was expected."""
    conn = make_conn()
    gaps = find_missing_trading_days(conn, "EMPTYCO", "2024-01-06", "2024-01-07")
    assert gaps.empty
    conn.close()


def test_find_missing_trading_days_gap_at_start_of_range_falls_back_to_start_date():
    """When the very first expected trading day is itself missing, there's
    no earlier row in the DB at all -- days_since_previous_available_date
    must fall back to start_date rather than crash on a None previous date."""
    conn = make_conn()
    symbol = "STARTGAP"
    for d in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]:  # 2024-01-01 missing
        insert_ohlcv(conn, symbol, [(d, 100, 101)])

    gaps = find_missing_trading_days(conn, symbol, "2024-01-01", "2024-01-05")

    assert len(gaps) == 1
    assert pd.Timestamp(gaps.iloc[0]["missing_date"]) == pd.Timestamp("2024-01-01")
    assert gaps.iloc[0]["days_since_previous_available_date"] == 0  # anchored to start_date itself
    conn.close()


def test_find_missing_trading_days_multiple_noncontiguous_gaps_each_anchored_to_own_previous_day():
    """Regression test: each gap must be anchored to its OWN previous
    available day, not a single anchor shared across every gap in the batch.

    Setup: weekdays 2024-01-01..2024-01-19 (15 confirmed NSE trading days,
    no holiday in range -- see test_get_nse_trading_days_excludes_weekends).
    Two separate single-day gaps: 2024-01-03 and 2024-01-16, with everything
    else present, including the unbroken run 2024-01-08..2024-01-15 sitting
    entirely between them.

    Hand-computed expected days_since_previous_available_date:
      2024-01-03: nearest prior available day is 2024-01-02 -> 1 day.
      2024-01-16: nearest prior available day is 2024-01-15 -> 1 day.

    Would catch: reverting to computing one previous_available_date from
    missing_days.min() and reusing it for every gap in the batch, which gave
    2024-01-16 an inflated (2024-01-16 - 2024-01-02) = 14 days instead of 1.
    """
    conn = make_conn()
    symbol = "MULTIGAP"
    present = [
        "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
        "2024-01-15", "2024-01-17", "2024-01-18", "2024-01-19",
    ]
    for d in present:
        insert_ohlcv(conn, symbol, [(d, 100, 101)])

    gaps = find_missing_trading_days(conn, symbol, "2024-01-01", "2024-01-19")
    gaps = gaps.sort_values("missing_date").reset_index(drop=True)
    assert len(gaps) == 2

    by_date = {
        pd.Timestamp(row["missing_date"]).strftime("%Y-%m-%d"): row["days_since_previous_available_date"]
        for _, row in gaps.iterrows()
    }
    assert by_date["2024-01-03"] == 1
    assert by_date["2024-01-16"] == 1  # each gap anchored to its OWN previous available day
    conn.close()


def test_log_gaps_dedupes_on_symbol_and_date():
    conn = make_conn()
    gaps_df = pd.DataFrame(
        {"symbol": ["AAA"], "missing_date": [pd.Timestamp("2024-01-03")],
         "days_since_previous_available_date": [1]}
    )
    log_gaps(conn, gaps_df)
    log_gaps(conn, gaps_df)  # re-logging the same gap must not duplicate

    rows = conn.execute("SELECT * FROM data_gaps_log").df()
    assert len(rows) == 1
    assert bool(rows.iloc[0]["resolved"]) is False
    conn.close()


def test_mark_resolved_flips_the_flag():
    conn = make_conn()
    gaps_df = pd.DataFrame(
        {"symbol": ["AAA"], "missing_date": [pd.Timestamp("2024-01-03")],
         "days_since_previous_available_date": [1]}
    )
    log_gaps(conn, gaps_df)
    mark_resolved(conn, "AAA", "2024-01-03")

    row = conn.execute(
        "SELECT resolved FROM data_gaps_log WHERE symbol='AAA' AND missing_date='2024-01-03'"
    ).fetchone()
    assert bool(row[0]) is True
    conn.close()

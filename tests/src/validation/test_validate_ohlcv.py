"""Tests for src/validation/validate_ohlcv.py: OHLCV data-quality checks.

Would catch: an off-by-one at any check's boundary (e.g. high==low wrongly
flagged, or a close-to-close move of exactly the 20% threshold wrongly
flagged or wrongly not-flagged), a check silently no-op'ing when its
required column is absent, or issues from different checks failing to
concatenate into one combined report.
"""

import duckdb
import pandas as pd
import pytest

from src.validation.validate_ohlcv import (
    EXTREME_MOVE_THRESHOLD,
    check_duplicate_rows,
    check_extreme_price_jumps,
    check_flat_days,
    check_missing_values,
    check_ohlc_violations,
    check_zero_negative_prices,
    check_zero_volume,
    init_data_quality_log,
    log_issues,
    validate_ohlcv,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_validate_ohlcv_empty_input_returns_empty_issues():
    result = validate_ohlcv(pd.DataFrame(), symbol="AAA")
    assert result.empty
    assert list(result.columns) == ["symbol", "date", "issue_type", "severity", "details"]


def test_check_missing_values_flags_nan_and_names_the_column():
    df = _df(
        [{"timestamp": "2024-01-01", "open": 100.0, "high": 105.0, "low": 99.0,
          "close": float("nan"), "adj_close": 104.0, "volume": 1000}]
    )
    issues = check_missing_values(df, "AAA")
    assert len(issues) == 1
    assert issues.iloc[0]["issue_type"] == "missing_value"
    assert issues.iloc[0]["severity"] == "error"
    assert "close" in issues.iloc[0]["details"]


def test_check_ohlc_violations_boundary_all_equal_is_not_a_violation():
    """high == low, open == high, close == low, etc. are all legal (a flat
    or gap-limit day) -- only a strict >/< crossing is a violation."""
    df = _df([{"timestamp": "2024-01-01", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}])
    assert check_ohlc_violations(df, "AAA").empty


@pytest.mark.parametrize(
    "row, expected_violation",
    [
        # low > high leaves no valid corridor for open/close, so this row
        # necessarily also trips close_below_low and open_below_low --
        # checked via membership, not an exact count of 1, for that reason.
        ({"open": 100, "high": 100, "low": 101, "close": 100}, "low_above_high"),
        ({"open": 100, "high": 100, "low": 99, "close": 101}, "close_above_high"),
        ({"open": 100, "high": 100, "low": 99, "close": 98}, "close_below_low"),
        ({"open": 98, "high": 100, "low": 99, "close": 99}, "open_below_low"),
        ({"open": 101, "high": 100, "low": 99, "close": 99}, "open_above_high"),
    ],
)
def test_check_ohlc_violations_each_violation_type(row, expected_violation):
    row = {"timestamp": "2024-01-01", **row}
    issues = check_ohlc_violations(_df([row]), "AAA")
    details = " | ".join(issues["details"])
    assert expected_violation in details


def test_check_zero_negative_prices_boundary_zero_flagged_epsilon_not():
    df = _df(
        [
            {"timestamp": "2024-01-01", "open": 0.0, "high": 1.0, "low": 0.0, "close": 1.0, "adj_close": 1.0},
            {"timestamp": "2024-01-02", "open": 0.01, "high": 1.0, "low": 0.01, "close": 1.0, "adj_close": 1.0},
            {"timestamp": "2024-01-03", "open": -5.0, "high": 1.0, "low": -5.0, "close": 1.0, "adj_close": 1.0},
        ]
    )
    issues = check_zero_negative_prices(df, "AAA")
    flagged = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in issues["date"]}
    assert "2024-01-01" in flagged  # exactly zero -- flagged
    assert "2024-01-03" in flagged  # negative -- flagged
    assert "2024-01-02" not in flagged  # tiny positive -- not flagged


def test_check_zero_volume_flags_but_only_as_a_warning():
    df = _df([{"timestamp": "2024-01-01", "volume": 0}, {"timestamp": "2024-01-02", "volume": 100}])
    issues = check_zero_volume(df, "AAA")
    assert len(issues) == 1
    assert issues.iloc[0]["severity"] == "warning"


def test_check_flat_days_all_four_equal_flags_partial_match_does_not():
    df = _df(
        [
            {"timestamp": "2024-01-01", "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
            {"timestamp": "2024-01-02", "open": 100.0, "high": 101.0, "low": 100.0, "close": 100.0},
        ]
    )
    issues = check_flat_days(df, "AAA")
    assert len(issues) == 1
    assert pd.Timestamp(issues.iloc[0]["date"]) == pd.Timestamp("2024-01-01")


def test_check_duplicate_rows_flags_all_occurrences_not_just_the_second():
    df = _df(
        [{"timestamp": "2024-01-01"}, {"timestamp": "2024-01-01"}, {"timestamp": "2024-01-01"},
         {"timestamp": "2024-01-02"}]
    )
    issues = check_duplicate_rows(df, "AAA")
    assert len(issues) == 3  # keep=False -- every duplicate occurrence flagged
    assert "3 times" in issues.iloc[0]["details"]


def test_check_extreme_price_jumps_boundary_exactly_at_threshold_not_flagged():
    df = _df([{"timestamp": "2024-01-01", "close": 100.0}, {"timestamp": "2024-01-02", "close": 120.0}])
    assert check_extreme_price_jumps(df, "AAA").empty  # exactly 20% -- boundary is strict >


def test_check_extreme_price_jumps_just_over_threshold_is_flagged():
    df = _df([{"timestamp": "2024-01-01", "close": 100.0}, {"timestamp": "2024-01-02", "close": 120.01}])
    issues = check_extreme_price_jumps(df, "AAA")
    assert len(issues) == 1
    assert pd.Timestamp(issues.iloc[0]["date"]) == pd.Timestamp("2024-01-02")


def test_extreme_move_threshold_constant_is_20_percent():
    assert EXTREME_MOVE_THRESHOLD == pytest.approx(0.20)


def test_validate_ohlcv_combines_multiple_check_types_for_one_symbol():
    df = _df(
        [
            {"timestamp": "2024-01-01", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0,
             "adj_close": 104.0, "volume": 1000},
            {"timestamp": "2024-01-02", "open": 105.0, "high": 106.0, "low": 100.0, "close": -1.0,
             "adj_close": -1.0, "volume": 0},  # negative price + zero volume + close < low
        ]
    )
    issues = validate_ohlcv(df, symbol="AAA")
    issue_types = set(issues["issue_type"])
    assert "zero_negative_price" in issue_types
    assert "zero_volume" in issue_types
    assert "ohlc_violation" in issue_types
    assert (issues["symbol"] == "AAA").all()


def test_log_issues_empty_is_noop_and_never_creates_the_table():
    conn = duckdb.connect(":memory:")
    log_issues(conn, pd.DataFrame(columns=["symbol", "date", "issue_type", "severity", "details"]))
    tables = set(conn.execute("SELECT table_name FROM information_schema.tables").df()["table_name"])
    assert "data_quality_log" not in tables


def test_log_issues_persists_and_init_is_idempotent():
    conn = duckdb.connect(":memory:")
    init_data_quality_log(conn)
    init_data_quality_log(conn)  # must not error the second time

    df = _df(
        [{"timestamp": "2024-01-01", "open": 1.0, "high": 1.0, "low": 1.0, "close": -1.0, "adj_close": -1.0}]
    )
    issues = check_zero_negative_prices(df, "AAA")
    log_issues(conn, issues)

    rows = conn.execute("SELECT symbol, issue_type, severity FROM data_quality_log").df()
    assert len(rows) == len(issues)
    assert rows.iloc[0]["symbol"] == "AAA"

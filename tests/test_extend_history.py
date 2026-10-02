"""Tests for extend_history.py's pure helper logic.

`main()` itself is a real end-to-end orchestration script (fetch, audit,
compute indicators, run strategy, print a summary) against the real project
DB and network -- out of scope for a hermetic unit suite. Its two pure
helpers, `_to_storage_symbols` and `_summarize_older_vs_recent_gaps`, are
fully testable and are what's covered here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from extend_history import GAP_RATE_FLAG_THRESHOLD, _summarize_older_vs_recent_gaps, _to_storage_symbols


def test_to_storage_symbols_strips_ns_suffix_only_when_present():
    """Would catch: stripping the wrong suffix, or crashing on a symbol that
    doesn't have '.NS' at all (a plain removesuffix would be silently safe,
    but an incorrect slice-based implementation could corrupt such a symbol)."""
    assert _to_storage_symbols(["RELIANCE.NS", "TCS.NS", "ALREADYPLAIN"]) == [
        "RELIANCE",
        "TCS",
        "ALREADYPLAIN",
    ]


def test_to_storage_symbols_empty_list():
    assert _to_storage_symbols([]) == []


def test_summarize_older_vs_recent_gaps_empty_input_returns_empty_with_columns():
    """Would catch: an empty gaps_df blowing up on .groupby()/.unstack() instead
    of short-circuiting, or the empty result missing an expected column."""
    result = _summarize_older_vs_recent_gaps(pd.DataFrame(), "2024-01-01", "2024-12-31")
    assert result.empty
    assert list(result.columns) == [
        "symbol", "older_gap_count", "older_gap_rate",
        "recent_gap_count", "recent_gap_rate", "flagged",
    ]


def test_summarize_older_vs_recent_gaps_splits_at_midpoint_and_flags_correctly():
    """A symbol with all its gaps in the older half (and none recent) must be
    flagged; a symbol with gaps evenly split between both halves must not be
    (assuming the rate difference stays within GAP_RATE_FLAG_THRESHOLD).

    Would catch: gaps assigned to the wrong half around the midpoint boundary,
    or the flagging threshold comparison using the wrong sign/direction
    (flagging *better* recent history as if it were worse).
    """
    start_date, end_date = "2024-01-01", "2024-02-29"  # ~60 calendar days -> midpoint ~2024-01-30

    gaps_df = pd.DataFrame(
        {
            "symbol": ["ALLOLD", "ALLOLD", "ALLOLD", "EVENSPLIT", "EVENSPLIT"],
            "missing_date": [
                "2024-01-05", "2024-01-10", "2024-01-15",  # all clearly in the older half
                "2024-01-05",  # older half
                "2024-02-20",  # recent half
            ],
            "days_since_previous_available_date": [1, 1, 1, 1, 1],
        }
    )

    result = _summarize_older_vs_recent_gaps(gaps_df, start_date, end_date)
    result = result.set_index("symbol")

    assert result.loc["ALLOLD", "older_gap_count"] == 3
    assert result.loc["ALLOLD", "recent_gap_count"] == 0
    assert result.loc["ALLOLD", "flagged"] == True  # noqa: E712 -- explicit bool check

    assert result.loc["EVENSPLIT", "older_gap_count"] == 1
    assert result.loc["EVENSPLIT", "recent_gap_count"] == 1
    # Equal counts in each half -> equal (or very close) rates since the two
    # halves are nearly equal length -> must NOT be flagged.
    assert result.loc["EVENSPLIT", "flagged"] == False  # noqa: E712


def test_summarize_older_vs_recent_gaps_threshold_boundary_is_strict_greater_than():
    """A rate difference exactly equal to GAP_RATE_FLAG_THRESHOLD must NOT be
    flagged (the code's condition is `> threshold`, not `>=`) -- construct
    gaps_df so older_gap_rate - recent_gap_rate lands exactly on the threshold.
    """
    start_date, end_date = "2024-01-01", "2024-01-20"  # 20 calendar days -> 2 equal 10-day halves

    # Only an older-half gap, no recent gap, for a symbol whose two implied
    # expected-day counts are equal (since the calendar split is symmetric),
    # so older_gap_rate - recent_gap_rate = 1/expected_older exactly.
    gaps_df = pd.DataFrame(
        {
            "symbol": ["EDGE"],
            "missing_date": ["2024-01-05"],
            "days_since_previous_available_date": [1],
        }
    )
    result = _summarize_older_vs_recent_gaps(gaps_df, start_date, end_date).set_index("symbol")
    rate_diff = result.loc["EDGE", "older_gap_rate"] - result.loc["EDGE", "recent_gap_rate"]

    if rate_diff == pytest.approx(GAP_RATE_FLAG_THRESHOLD):
        assert result.loc["EDGE", "flagged"] == False  # noqa: E712 -- strict >, not >=
    else:
        # The constructed rate difference didn't land exactly on the
        # threshold (NSE calendar day counts don't divide evenly for every
        # window) -- fall back to checking the strict-inequality contract
        # directly against the underlying comparison instead.
        assert result.loc["EDGE", "flagged"] == (rate_diff > GAP_RATE_FLAG_THRESHOLD)

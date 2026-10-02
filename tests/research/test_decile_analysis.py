"""Correctness tests for research/decile_analysis.py.

bucket_by_decile/summarize_deciles are pure functions on small hand-picked
panels; plot_decile_returns is checked only for "does it write a file".
"""

from __future__ import annotations

import pandas as pd
import pytest

from research.decile_analysis import bucket_by_decile, plot_decile_returns, summarize_deciles


def test_bucket_by_decile_hand_computed_values():
    """4 stocks, 1 date, n_buckets=2: signal=[1,2,3,4] splits into bucket1
    (AAA,BBB) and bucket2 (CCC,DDD) via qcut. Would catch a wrong bucket
    direction (e.g. 1 = highest instead of lowest signal value, which the
    docstring explicitly defines), or the mean_fwd_return aggregation being
    computed over the wrong rows.
    """
    dates = pd.Series(["2024-01-01"] * 4)
    symbols = pd.Series(["AAA", "BBB", "CCC", "DDD"])
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    fwd = pd.Series([0.1, 0.2, -0.1, 0.3])

    result = bucket_by_decile(signal, fwd, dates, symbols, n_buckets=2)
    result = result.set_index("bucket")

    assert result.loc[1, "mean_fwd_return"] == pytest.approx(0.15)  # mean(0.1, 0.2)
    assert result.loc[1, "n_stocks"] == 2
    assert result.loc[2, "mean_fwd_return"] == pytest.approx(0.1)  # mean(-0.1, 0.3)
    assert result.loc[2, "n_stocks"] == 2


def test_bucket_by_decile_skips_date_with_too_few_stocks():
    """A date with fewer valid stocks than n_buckets can't form that many
    groups -- must be skipped, not crash or silently form fewer/degenerate
    buckets."""
    dates = pd.Series(["2024-01-01", "2024-01-01", "2024-01-01"])
    symbols = pd.Series(["AAA", "BBB", "CCC"])
    signal = pd.Series([1.0, 2.0, 3.0])
    fwd = pd.Series([0.1, 0.2, 0.3])

    result = bucket_by_decile(signal, fwd, dates, symbols, n_buckets=5)
    assert result.empty


def test_bucket_by_decile_all_identical_signal_produces_no_rows():
    """Every stock has the exact identical signal value on a date -- qcut
    can't rank them into distinct buckets, so that date must contribute no
    rows at all (not a spurious single mega-bucket).

    NOTE for reviewer: bucket_by_decile.py wraps the qcut call in
    `try/except ValueError` with a comment claiming that's what catches this
    case ("All signal values identical that day -> qcut can't form any
    bins"). On the pandas version actually installed here (3.0.5),
    pd.qcut(..., duplicates='drop') does NOT raise ValueError for an
    all-identical input -- it returns an all-NaN bucket-label Series instead,
    verified directly:
        pd.qcut(pd.Series([1.0,1.0,1.0,1.0]), q=2, labels=False, duplicates='drop')
        -> [nan, nan, nan, nan], no exception
    The end result observed below is still correct (this date is skipped),
    but it happens via a DIFFERENT, undocumented mechanism than the one the
    code comments claim: `group.assign(bucket=bucket + 1)` produces an
    all-NaN bucket column, and `.groupby("bucket")` then drops NaN keys by
    default, so the for-loop over groups never executes. The `except
    ValueError` branch is dead code for this input on this pandas version --
    functionally harmless right now, but fragile (relying on groupby's
    default NaN-key-dropping rather than the exception handler the comment
    describes), and worth a second look rather than silently "fixed" here.
    """
    dates = pd.Series(["2024-01-01"] * 4)
    symbols = pd.Series(["AAA", "BBB", "CCC", "DDD"])
    signal = pd.Series([1.0, 1.0, 1.0, 1.0])
    fwd = pd.Series([0.1, 0.2, -0.1, 0.3])

    result = bucket_by_decile(signal, fwd, dates, symbols, n_buckets=2)
    assert result.empty


def test_bucket_by_decile_drops_nan_rows_before_bucketing():
    """A NaN signal or forward return must be excluded before bucketing, not
    propagate a NaN bucket or shrink the group below n_buckets unexpectedly."""
    dates = pd.Series(["2024-01-01"] * 5)
    symbols = pd.Series(["AAA", "BBB", "CCC", "DDD", "EEE"])
    signal = pd.Series([1.0, 2.0, float("nan"), 3.0, 4.0])
    fwd = pd.Series([0.1, 0.2, 0.5, -0.1, 0.3])

    result = bucket_by_decile(signal, fwd, dates, symbols, n_buckets=2)
    assert result["n_stocks"].sum() == 4  # EEE's NaN row (CCC here) excluded


def test_summarize_deciles_hand_computed_spread():
    """From the hand-computed bucket_df above: bucket1 mean=0.15 (n=2),
    bucket2 mean=0.1 (n=2) -> spread = bucket2 - bucket1 = -0.05. Would catch
    the spread direction being inverted (top - bottom vs bottom - top)."""
    bucket_df = pd.DataFrame(
        [
            {"date": "2024-01-01", "bucket": 1, "mean_fwd_return": 0.1, "n_stocks": 2},
            {"date": "2024-01-01", "bucket": 2, "mean_fwd_return": 0.2, "n_stocks": 2},
            {"date": "2024-01-02", "bucket": 1, "mean_fwd_return": 0.2, "n_stocks": 2},
            {"date": "2024-01-02", "bucket": 2, "mean_fwd_return": 0.0, "n_stocks": 2},
        ]
    )
    result = summarize_deciles(bucket_df)

    # bucket1 mean across dates = mean(0.1, 0.2) = 0.15; bucket2 = mean(0.2, 0.0) = 0.1
    assert result.loc[1, "mean_fwd_return"] == pytest.approx(0.15)
    assert result.loc[2, "mean_fwd_return"] == pytest.approx(0.1)
    assert result.loc[1, "n_observations"] == 4
    assert result.loc["spread", "mean_fwd_return"] == pytest.approx(0.1 - 0.15)


def test_summarize_deciles_empty_input():
    """Would catch a groupby-on-empty-frame crash."""
    result = summarize_deciles(pd.DataFrame(columns=["date", "bucket", "mean_fwd_return", "n_stocks"]))
    assert result.empty
    assert list(result.columns) == ["mean_fwd_return", "n_observations"]


def test_plot_decile_returns_writes_file_and_excludes_spread_row(tmp_path):
    """Smoke test: writes a PNG, and doesn't error when the 'spread' row is
    present (it should be excluded from the plotted bars, not passed to a
    string-cast x-axis that would visually merge it in as bucket 'spread')."""
    summary_df = pd.DataFrame(
        {"mean_fwd_return": [0.15, 0.1, -0.05], "n_observations": [4, 4, 8]},
        index=[1, 2, "spread"],
    )
    out = plot_decile_returns(summary_df, tmp_path / "nested" / "deciles.png")
    assert out.exists()
    assert out.stat().st_size > 0

    empty_summary = pd.DataFrame(columns=["mean_fwd_return", "n_observations"])
    out2 = plot_decile_returns(empty_summary, tmp_path / "empty.png")
    assert out2.exists()

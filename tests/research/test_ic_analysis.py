"""Correctness tests for research/ic_analysis.py.

calculate_ic/summarize_ic are pure functions on small hand-picked series;
plot_ic_over_time is checked only for "does it write a file" via tmp_path
(matplotlib is already forced to the Agg backend by the module itself).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.ic_analysis import calculate_ic, plot_ic_over_time, summarize_ic


def test_calculate_ic_invalid_method_raises():
    """Would catch a typo'd method string silently falling through to Pearson."""
    with pytest.raises(ValueError, match="spearman.*pearson"):
        calculate_ic(pd.Series([1.0]), pd.Series([1.0]), pd.Series(["2024-01-01"]), pd.Series(["AAA"]), method="kendall")


def test_calculate_ic_spearman_vs_pearson_differ_on_nonlinear_monotonic_data():
    """signal=[1,2,3,4] vs fwd_return=[1,4,9,16] is perfectly monotonic but
    nonlinear: Spearman must be exactly 1.0, Pearson must be < 1.0. Would
    catch 'spearman' silently computing a plain Pearson correlation instead
    of ranking first (the module's own docstring calls out that it
    implements Spearman as rank-then-Pearson to avoid a scipy dependency --
    this test is exactly the regression that formula could introduce).
    """
    dates = pd.Series(["2024-01-01"] * 4)
    symbols = pd.Series(["AAA", "BBB", "CCC", "DDD"])
    signal = pd.Series([1.0, 2.0, 3.0, 4.0])
    fwd = pd.Series([1.0, 4.0, 9.0, 16.0])

    spearman = calculate_ic(signal, fwd, dates, symbols, method="spearman")
    pearson = calculate_ic(signal, fwd, dates, symbols, method="pearson")

    assert spearman["ic"].iloc[0] == pytest.approx(1.0)
    assert pearson["ic"].iloc[0] == pytest.approx(0.9843740386976972, rel=1e-9)
    assert pearson["ic"].iloc[0] < spearman["ic"].iloc[0]


def test_calculate_ic_fewer_than_3_stocks_is_nan():
    """A date with only 2 valid (signal, forward_return) pairs is too few to
    correlate meaningfully -- the function documents a hard n_stocks < 3
    floor. Would catch that floor being silently dropped (correlating on 2
    points always gives +-1.0 or NaN, which looks deceptively confident).
    """
    dates = pd.Series(["2024-01-01", "2024-01-01"])
    symbols = pd.Series(["AAA", "BBB"])
    result = calculate_ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), dates, symbols)
    assert result["n_stocks"].iloc[0] == 2
    assert pd.isna(result["ic"].iloc[0])


def test_calculate_ic_drops_nan_rows_before_correlating():
    """A NaN signal or forward return on some rows must be excluded from
    that date's stock count and correlation, not propagate NaN into the
    whole day's result or crash the correlation."""
    dates = pd.Series(["2024-01-01"] * 4)
    symbols = pd.Series(["AAA", "BBB", "CCC", "DDD"])
    signal = pd.Series([1.0, 2.0, np.nan, 4.0])
    fwd = pd.Series([1.0, 4.0, 9.0, 16.0])
    result = calculate_ic(signal, fwd, dates, symbols)
    assert result["n_stocks"].iloc[0] == 3  # CCC's NaN row excluded
    assert not pd.isna(result["ic"].iloc[0])


def test_summarize_ic_hand_computed_values():
    """Hand-computed mean/std/IR/t-stat on a 5-day IC series.

    Would catch a wrong annualization factor (sqrt(252)), a population
    (ddof=0) instead of sample std, or a t-stat formula missing the sqrt(n).
    """
    ic_df = pd.DataFrame({"ic": [0.02, -0.01, 0.03, 0.05, -0.02]})
    result = summarize_ic(ic_df)

    assert result["n_days"] == 5
    assert result["mean_ic"] == pytest.approx(0.014, rel=1e-9)
    assert result["std_ic"] == pytest.approx(0.028809720581775868, rel=1e-9)
    assert result["ic_ir"] == pytest.approx(7.7141709687392686, rel=1e-9)
    assert result["t_stat"] == pytest.approx(1.086610735988866, rel=1e-9)
    assert result["pct_positive_days"] == pytest.approx(0.6)


def test_summarize_ic_empty_input_returns_all_nan_zero_days():
    """No valid IC observations at all -- would catch a ZeroDivisionError or
    a fabricated 0.0 mean instead of the documented all-NaN/n_days=0 result."""
    result = summarize_ic(pd.DataFrame({"ic": [np.nan, np.nan]}))
    assert result["n_days"] == 0
    assert pd.isna(result["mean_ic"])
    assert pd.isna(result["ic_ir"])
    assert pd.isna(result["t_stat"])


def test_summarize_ic_zero_variance_ic_series_is_nan_ir_not_crash():
    """Every day has the exact same IC value -- std_ic is 0, so ic_ir/t_stat
    would divide by zero. Would catch a raw ZeroDivisionError/inf instead of
    the documented NaN guard (`if std_ic > 0 ... else nan`)."""
    result = summarize_ic(pd.DataFrame({"ic": [0.03, 0.03, 0.03]}))
    assert result["n_days"] == 3
    assert result["mean_ic"] == pytest.approx(0.03)
    assert result["std_ic"] == pytest.approx(0.0)
    assert pd.isna(result["ic_ir"])
    assert pd.isna(result["t_stat"])


def test_plot_ic_over_time_writes_file(tmp_path):
    """Smoke test: the plot function actually writes a PNG to the requested
    path (including creating parent directories), for both the normal and
    all-NaN-IC (empty-after-dropna) cases."""
    ic_df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5, freq="D"), "ic": [0.01, -0.02, 0.03, np.nan, 0.02]}
    )
    out = plot_ic_over_time(ic_df, tmp_path / "nested" / "ic.png")
    assert out.exists()
    assert out.stat().st_size > 0

    empty_ic_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="D"), "ic": [np.nan] * 3})
    out2 = plot_ic_over_time(empty_ic_df, tmp_path / "empty_ic.png")
    assert out2.exists()

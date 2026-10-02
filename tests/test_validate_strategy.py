"""Tests for validate_strategy.py's in-sample/out-of-sample split and grid-search wiring.

This module exists to prevent the classic multiple-comparisons trap (see its
own module docstring): a parameter grid always produces a "best" row even
with no real edge, so the in-sample/out-of-sample split must be leak-free.
These tests focus on that split's exact boundary arithmetic, and on
`run_parameter_grid`/`run_out_of_sample_test` wiring the shared backtest
engine correctly over a restricted date window.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.registry import get_strategy
from tests.helpers import insert_ohlcv, make_conn, seed_active_universe

import backtest as bt
from validate_strategy import (
    _active_storage_symbols,
    _backtest_signals,
    _filter_to_range,
    _load_ohlcv_history,
    compute_in_sample_split,
    run_out_of_sample_test,
    run_parameter_grid,
)


def _seed_days(conn, symbol, n, start="2024-01-01"):
    """Insert n consecutive daily rows (flat price 100) so only the *count* of
    distinct trading days matters, not their values."""
    dates = pd.date_range(start, periods=n, freq="D")
    insert_ohlcv(conn, symbol, [(d.strftime("%Y-%m-%d"), 100.0, 100.0) for d in dates])
    return [d.strftime("%Y-%m-%d") for d in dates]


def test_compute_in_sample_split_exact_boundary():
    """80/20 split over 10 known trading days lands exactly where hand math says.

    Would catch: an off-by-one that leaks the first out-of-sample day into
    the in-sample window (the actual overfitting bug this module exists to
    prevent), or a split computed from calendar days instead of the actual
    distinct trading days present in ohlcv_data.
    """
    conn = make_conn()
    dates = _seed_days(conn, "SPLITCO", 10)

    full_start, in_sample_end, out_of_sample_start, full_end = compute_in_sample_split(
        conn, in_sample_fraction=0.8
    )

    # Hand-computed: split_idx = int(10 * 0.8) = 8 -> in-sample is dates[0:8],
    # out-of-sample starts at dates[8].
    assert full_start == dates[0]
    assert in_sample_end == dates[7]
    assert out_of_sample_start == dates[8]
    assert full_end == dates[9]
    # The two windows must never overlap or share a day.
    assert pd.Timestamp(out_of_sample_start) > pd.Timestamp(in_sample_end)


def test_compute_in_sample_split_clamps_extreme_fractions():
    """An in_sample_fraction near 0 or 1 still leaves >=1 day on each side.

    Would catch: a fraction of 0.99 (or 0.01) producing a zero-length
    out-of-sample (or in-sample) window, silently making the "held-out" test
    meaningless.
    """
    conn = make_conn()
    dates = _seed_days(conn, "CLAMPCO", 10)

    _, in_sample_end, out_of_sample_start, _ = compute_in_sample_split(conn, in_sample_fraction=0.99)
    assert in_sample_end == dates[8]
    assert out_of_sample_start == dates[9]  # exactly 1 day held out, not 0

    _, in_sample_end, out_of_sample_start, _ = compute_in_sample_split(conn, in_sample_fraction=0.01)
    assert in_sample_end == dates[0]  # exactly 1 day in-sample, not 0
    assert out_of_sample_start == dates[1]


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.1])
def test_compute_in_sample_split_rejects_out_of_range_fraction(bad_fraction):
    """in_sample_fraction outside the open interval (0, 1) is rejected up front."""
    conn = make_conn()
    _seed_days(conn, "BADCO", 10)
    with pytest.raises(ValueError):
        compute_in_sample_split(conn, in_sample_fraction=bad_fraction)


def test_compute_in_sample_split_no_data_raises():
    """No OHLCV data at all raises a clear RuntimeError rather than crashing
    inside pandas on an empty index (e.g. dates.iloc[0] on an empty Series)."""
    conn = make_conn()
    with pytest.raises(RuntimeError):
        compute_in_sample_split(conn)


def test_filter_to_range_boundary_inclusive_both_ends():
    """_filter_to_range keeps rows exactly on start_date/end_date, not just strictly inside.

    Would catch: an accidental strict inequality that drops a signal dated
    exactly on the window's first or last day.
    """
    signals_df = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-10", "2024-01-11"]),
            "signal_type": ["BUY", "BUY", "SELL", "SELL"],
        }
    )
    filtered = _filter_to_range(signals_df, "2024-01-05", "2024-01-10")
    assert sorted(filtered["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-10"]


def test_filter_to_range_empty_input_returned_unchanged():
    """An empty signals_df short-circuits rather than erroring on a missing 'date' column."""
    empty = pd.DataFrame()
    assert _filter_to_range(empty, "2024-01-01", "2024-01-31") is empty


def test_backtest_signals_empty_inputs_yield_zero_metrics():
    """Empty signals or empty prices must not crash _backtest_signals -- it
    should fall through to calculate_metrics's documented all-zero result.

    Would catch: an unguarded .pivot()/groupby() on an empty DataFrame
    raising instead of producing a clean zero-trade metrics dict.
    """
    metrics = _backtest_signals(pd.DataFrame(), pd.DataFrame())
    assert metrics["total_trades"] == 0
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown_pct"] == 0.0


def test_run_parameter_grid_matches_manual_pipeline():
    """run_parameter_grid's row for one param combo matches running the same
    generate_signals -> filter -> backtest steps manually, one level up.

    Would catch: run_parameter_grid loading price data over the wrong date
    range (e.g. full history's P&L instead of [start_date, end_date]),
    signals from outside the requested window leaking into the backtest, or
    the returned row's param columns not matching what was actually run.
    """
    conn = make_conn()
    symbol = "GRIDCO"
    seed_active_universe(conn, [symbol])

    # 20 days: a decline (warms up the SMAs with no crossover) then a sharp
    # rise, guaranteeing the fast SMA(3) crosses above the slow SMA(5) at
    # some point in the second half.
    rows = []
    price = 200.0
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    for i, d in enumerate(dates):
        price = price - 3 if i < 10 else price + 6
        rows.append((d.strftime("%Y-%m-%d"), price, price))
    insert_ohlcv(conn, symbol, rows)

    strategy_name = "sma_crossover"
    param_grid = [{"fast_window": 3, "slow_window": 5}]
    start_date, end_date = "2024-01-01", "2024-01-20"

    grid_df = run_parameter_grid(conn, strategy_name, param_grid, start_date, end_date)
    assert len(grid_df) == 1
    row = grid_df.iloc[0]
    assert row["fast_window"] == 3
    assert row["slow_window"] == 5
    assert row["total_trades"] >= 1  # the constructed rise must have produced a crossover BUY

    # Independent re-derivation of the same pipeline, to cross-check wiring:
    symbols = _active_storage_symbols(conn)
    assert symbols == [symbol]
    history = _load_ohlcv_history(conn, symbols, end_date)
    price_df = bt._load_daily_prices(conn, symbols, start_date, end_date)
    strategy = get_strategy(strategy_name)(fast_window=3, slow_window=5)
    signals_df = _filter_to_range(strategy.generate_signals(history), start_date, end_date)
    expected_metrics = _backtest_signals(signals_df, price_df)

    assert row["total_trades"] == expected_metrics["total_trades"]
    if pd.notna(expected_metrics["cagr"]):
        assert row["cagr"] == pytest.approx(expected_metrics["cagr"], abs=1e-9)
    if pd.notna(expected_metrics["sharpe_ratio"]):
        assert row["sharpe_ratio"] == pytest.approx(expected_metrics["sharpe_ratio"], abs=1e-9)


def test_run_out_of_sample_test_only_uses_its_own_window():
    """run_out_of_sample_test's metrics must come only from signals/prices
    inside [out_of_sample_start, out_of_sample_end] -- not the full history.

    Would catch: forgetting to filter generate_signals's output to the
    out-of-sample window, letting an in-sample-window trade's PnL leak into
    what's supposed to be a held-out result.
    """
    conn = make_conn()
    symbol = "OOSCO"
    seed_active_universe(conn, [symbol])

    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    price = 100.0
    for i, d in enumerate(dates):
        # A crossover-triggering move early (in-sample) and another later
        # (out-of-sample), each big enough to guarantee a signal.
        if i == 6:
            price += 40
        if i == 22:
            price -= 40
        rows.append((d.strftime("%Y-%m-%d"), price, price))
    insert_ohlcv(conn, symbol, rows)

    oos_start, oos_end = "2024-01-21", "2024-01-30"
    metrics = run_out_of_sample_test(
        conn, "sma_crossover", oos_start, oos_end, fast_window=3, slow_window=5
    )

    # Independently recompute using only the out-of-sample window, same as above.
    symbols = _active_storage_symbols(conn)
    history = _load_ohlcv_history(conn, symbols, oos_end)
    price_df = bt._load_daily_prices(conn, symbols, oos_start, oos_end)
    strategy = get_strategy("sma_crossover")(fast_window=3, slow_window=5)
    signals_df = _filter_to_range(strategy.generate_signals(history), oos_start, oos_end)
    expected = _backtest_signals(signals_df, price_df)

    assert metrics["total_trades"] == expected["total_trades"]

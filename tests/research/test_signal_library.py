"""Correctness tests for research/signal_library.py.

Each signal is a pure function on a small synthetic OHLCV panel with
hand-computable expected values -- no DB, no network. Docstrings say what
bug each test would catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signal_library import (
    SignalSpec,
    available_signals,
    bb_position,
    cross_sectional_rank_momentum,
    get_signal,
    momentum,
    register_signal,
    rsi_level,
    volatility,
    volume_weighted_momentum,
)
from research.signal_library import _compute_rsi


def _panel(symbol_series: dict[str, list[float]], volumes: dict[str, list[float]] | None = None) -> pd.DataFrame:
    """Build a multi-symbol adj_close (+ optional volume) panel sharing one date axis."""
    n = len(next(iter(symbol_series.values())))
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    frames = []
    for symbol, closes in symbol_series.items():
        frame = pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})
        if volumes is not None:
            frame["volume"] = volumes[symbol]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Registry mechanics (same pattern as strategies.registry)
# ---------------------------------------------------------------------------


def test_available_signals_includes_all_builtins():
    """Would catch a signal module failing to register itself (e.g. a typo in
    the @register_signal decorator call, or an import-time exception silently
    swallowed elsewhere)."""
    names = available_signals()
    assert names == sorted(names)
    for expected in (
        "momentum",
        "volume_weighted_momentum",
        "rsi_level",
        "bb_position",
        "volatility",
        "cross_sectional_rank_momentum",
    ):
        assert expected in names


def test_get_signal_unknown_name_raises_and_lists_available():
    """Would catch a KeyError message that doesn't actually list what's registered."""
    with pytest.raises(KeyError, match="momentum"):
        get_signal("definitely_not_a_real_signal")


def test_register_signal_same_function_twice_is_not_an_error():
    """Re-importing a module that re-runs its own @register_signal decorators
    (e.g. via a reload) must not raise -- only a genuinely different function
    registered under an existing name should."""

    @register_signal("test_dup_signal", default_params={"x": 1})
    def _f(df, params):
        return df["adj_close"]

    # Re-registering the exact same function object under the same name is a no-op.
    register_signal("test_dup_signal", default_params={"x": 1})(_f)

    with pytest.raises(ValueError, match="already registered"):
        @register_signal("test_dup_signal", default_params={"x": 1})
        def _g(df, params):
            return df["adj_close"]


def test_signal_spec_call_merges_params_over_defaults():
    """Would catch a merge-direction bug (defaults overriding explicit params
    instead of the other way around)."""
    calls = []

    def _compute(df, params):
        calls.append(dict(params))
        return pd.Series([0.0] * len(df))

    spec = SignalSpec(name="probe", compute=_compute, default_params={"window": 20, "num_std": 2.0})
    df = pd.DataFrame({"adj_close": [1.0, 2.0]})

    spec(df)  # no override -> defaults pass through untouched
    assert calls[-1] == {"window": 20, "num_std": 2.0}

    spec(df, {"window": 5})  # explicit override wins
    assert calls[-1] == {"window": 5, "num_std": 2.0}


# ---------------------------------------------------------------------------
# _compute_rsi
# ---------------------------------------------------------------------------


def test_compute_rsi_hand_computed_values():
    """Hand-computed RSI(period=2) on a 5-point series.

    Would catch a wrong gain/loss split, a wrong rolling window/min_periods,
    or an inverted RSI formula (100-RSI instead of RSI).
    """
    series = pd.Series([100.0, 102.0, 101.0, 105.0, 103.0])
    rsi = _compute_rsi(series, period=2)

    assert rsi.iloc[0:2].isna().all()  # not enough history yet
    assert rsi.iloc[2] == pytest.approx(66.66666666666666)
    assert rsi.iloc[3] == pytest.approx(80.0)
    assert rsi.iloc[4] == pytest.approx(66.66666666666666)


def test_compute_rsi_monotonic_boundaries():
    """A monotonically rising series has zero losses -> RSI pinned at exactly
    100 once warmed up; a monotonically falling series has zero gains ->
    RSI pinned at exactly 0. Would catch an inverted or off-by-one formula
    that doesn't saturate correctly at these extremes."""
    rising = _compute_rsi(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), period=2)
    assert rising.iloc[2:].eq(100.0).all()

    falling = _compute_rsi(pd.Series([5.0, 4.0, 3.0, 2.0, 1.0]), period=2)
    assert falling.iloc[2:].eq(0.0).all()


def test_compute_rsi_flat_series_is_100_not_nan():
    """A perfectly flat series has avg_gain=0 AND avg_loss=0 (0/0 for the raw
    ratio) -- the implementation's explicit `.where(avg_loss != 0, 100.0)`
    guard means this resolves to exactly 100, not NaN. Documents this exact,
    slightly non-obvious documented tie-breaking choice so a future formula
    change that flips it is caught."""
    flat = _compute_rsi(pd.Series([5.0, 5.0, 5.0, 5.0]), period=2)
    assert flat.iloc[2:].eq(100.0).all()


# ---------------------------------------------------------------------------
# momentum / volume_weighted_momentum
# ---------------------------------------------------------------------------


def test_momentum_hand_computed_and_symbol_isolated():
    """Two symbols on the same date axis with different price paths and a
    momentum window of 3. Would catch a wrong shift direction/magnitude, or
    -- since this uses groupby("symbol").transform -- a symbol-boundary leak
    where symbol B's momentum gets computed using symbol A's price a few rows
    away (a real bug class in panel .shift() operations).
    """
    df = _panel(
        {
            "AAA": [100.0, 102.0, 104.0, 106.0, 110.0],
            "BBB": [50.0, 50.0, 50.0, 50.0, 60.0],
        }
    )
    result = momentum(df, {"window": 3})
    df = df.assign(momentum=result)

    aaa = df[df["symbol"] == "AAA"]["momentum"].reset_index(drop=True)
    bbb = df[df["symbol"] == "BBB"]["momentum"].reset_index(drop=True)

    assert aaa.iloc[:3].isna().all()
    assert aaa.iloc[3] == pytest.approx(106.0 / 100.0 - 1)
    assert aaa.iloc[4] == pytest.approx(110.0 / 102.0 - 1)

    assert bbb.iloc[:3].isna().all()
    assert bbb.iloc[3] == pytest.approx(50.0 / 50.0 - 1)  # 0.0 -- not AAA's 0.06
    assert bbb.iloc[4] == pytest.approx(60.0 / 50.0 - 1)


def test_volume_weighted_momentum_missing_volume_column_raises():
    """Would catch the required-columns guard silently disappearing (e.g. a
    refactor that reads df.get("volume") instead of indexing it)."""
    df = _panel({"AAA": [100.0, 101.0, 102.0]})  # no volume column
    with pytest.raises(ValueError, match="volume_weighted_momentum requires columns"):
        volume_weighted_momentum(df, {"window": 2})


def test_volume_weighted_momentum_hand_computed():
    """price_momentum * (volume / rolling_avg_volume), window=2.

    Would catch the volume scaling being applied as a division instead of a
    multiplication (the module's own docstring calls this out explicitly as
    an intentional choice, since dividing would dampen high-volume moves
    instead of amplifying them).
    """
    df = _panel(
        {"AAA": [100.0, 110.0, 121.0, 100.0]},
        volumes={"AAA": [1000.0, 1000.0, 3000.0, 1000.0]},
    )
    result = volume_weighted_momentum(df, {"window": 2})

    # index2: price_momentum = 121/100 - 1 = 0.21; relative_volume = 3000 / mean([1000,3000]) = 1.5
    assert result.iloc[2] == pytest.approx(0.21 * 1.5)
    # index3: price_momentum = 100/110 - 1; relative_volume = 1000 / mean([3000,1000]) = 0.5
    assert result.iloc[3] == pytest.approx((100.0 / 110.0 - 1) * 0.5)


# ---------------------------------------------------------------------------
# rsi_level
# ---------------------------------------------------------------------------


def test_rsi_level_matches_compute_rsi_reindexed():
    """rsi_level is just _compute_rsi run per-symbol and reindexed back onto
    the input frame -- would catch that reindexing losing alignment (e.g.
    silently reindexing by position instead of by the original index) when
    the input isn't already sorted by symbol/date."""
    df = _panel({"AAA": [100.0, 102.0, 101.0, 105.0, 103.0]})
    # Shuffle row order -- rsi_level must sort internally and reindex back correctly.
    shuffled = df.sample(frac=1.0, random_state=0)

    result = rsi_level(shuffled, {"period": 2})
    aligned = pd.DataFrame({"date": shuffled["date"], "rsi": result}).sort_values("date")

    expected = _compute_rsi(df["adj_close"], period=2)
    assert aligned["rsi"].reset_index(drop=True).equals(expected.reset_index(drop=True)) or np.allclose(
        aligned["rsi"].reset_index(drop=True).fillna(-999),
        expected.reset_index(drop=True).fillna(-999),
    )


# ---------------------------------------------------------------------------
# bb_position
# ---------------------------------------------------------------------------


def test_bb_position_hand_computed_value():
    """window=3, num_std=2 on prices [10,20,30]: mean=20, sample std=10
    (ddof=1, pandas' rolling default) -> upper=40, lower=0 -> position at the
    3rd bar = (30-0)/(40-0) = 0.75. Would catch a wrong std ddof, a swapped
    upper/lower, or num_std not being applied.
    """
    df = _panel({"AAA": [10.0, 20.0, 30.0]})
    result = bb_position(df, {"window": 3, "num_std": 2.0})
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(0.75)


def test_bb_position_zero_variance_window_is_nan_not_crash():
    """A perfectly flat window has std=0 -> upper==lower -> the position
    formula divides by zero. pandas float division gives NaN for 0/0, not an
    exception -- would catch a change that turns this into a crash or a
    fabricated finite value (e.g. defaulting to 0.5)."""
    df = _panel({"AAA": [10.0, 10.0, 10.0]})
    result = bb_position(df, {"window": 3, "num_std": 2.0})
    assert pd.isna(result.iloc[2])


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------


def test_volatility_hand_computed_value():
    """Rolling 3-day std (ddof=1) of daily returns on a hand-picked series.

    Returns are [NaN, +10%, -10%, +10%(approx)]; std of the last three
    non-NaN returns [0.1, -0.1, 0.1] (ddof=1) = 0.11547005383792519. Would
    catch using price levels instead of returns, or a population (ddof=0)
    std instead of sample std.
    """
    df = _panel({"AAA": [100.0, 110.0, 99.0, 108.9]})
    result = volatility(df, {"window": 3})
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(0.11547005383792519, rel=1e-9)


# ---------------------------------------------------------------------------
# cross_sectional_rank_momentum
# ---------------------------------------------------------------------------


def test_cross_sectional_rank_momentum_known_ordering():
    """3 symbols, one shared date, momentum window=1 (so it's just the prior
    day's return) with a known ranking: BBB < AAA < CCC -> percentile ranks
    1/3, 2/3, 1.0. Would catch ranking within the wrong axis (e.g. ranking
    across dates for one symbol instead of across symbols within one date).
    """
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "date": [dates[0], dates[1], dates[0], dates[1], dates[0], dates[1]],
            # 1-day momentum on day2: AAA +10%, BBB -10%, CCC +30%
            "adj_close": [100.0, 110.0, 100.0, 90.0, 100.0, 130.0],
        }
    )
    result = cross_sectional_rank_momentum(df, {"window": 1})
    df = df.assign(rank=result)
    day2 = df[df["date"] == dates[1]].set_index("symbol")["rank"]

    assert day2["BBB"] == pytest.approx(1.0 / 3.0)
    assert day2["AAA"] == pytest.approx(2.0 / 3.0)
    assert day2["CCC"] == pytest.approx(1.0)


def test_cross_sectional_rank_momentum_tie_gets_average_rank():
    """Two symbols with IDENTICAL momentum on the same date -- pandas'
    rank(pct=True) default tie-break is 'average': for 2 fully-tied values,
    the average rank is (1+2)/2 = 1.5, so pct = 1.5/2 = 0.75 for BOTH -- the
    key property being tested is that they land on the SAME percentile, not
    an arbitrary stable-sort winner getting a strictly higher rank than the
    loser (0.75/0.75, not e.g. 0.5/1.0 or 1.0/0.5)."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "date": [dates[0], dates[1], dates[0], dates[1]],
            "adj_close": [100.0, 110.0, 100.0, 110.0],  # identical +10% move
        }
    )
    result = cross_sectional_rank_momentum(df, {"window": 1})
    df = df.assign(rank=result)
    day2 = df[df["date"] == dates[1]].set_index("symbol")["rank"]

    assert day2["AAA"] == pytest.approx(0.75)
    assert day2["BBB"] == pytest.approx(0.75)
    assert day2["AAA"] == day2["BBB"]

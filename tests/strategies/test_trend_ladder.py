"""Correctness tests for strategies/trend_ladder.py.

The strategy's default config needs ~200 days of warm-up (EMA 200) to reach
any signal at all, which isn't hand-traceable. Every generate_signals() test
here instead overrides the EMA/ADX/volume periods down to small values (2/3/
4/5/6, still strictly increasing as `TrendLadderConfig.validate()` requires)
so a short, fully hand-traced synthetic series is enough to pin down exact
trigger days. Every number below was cross-checked against the actual EWM
math (span-based EMA with adjust=False starts at the first value and is
invariant under a flat run, alpha=2/(span+1)) before being hardcoded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.trend_ladder import TrendLadderConfig, TrendLadderStrategy, _compute_adx

# Small periods so a handful of synthetic rows are enough to warm everything
# up; strictly increasing per TrendLadderConfig.validate()'s EMA-ordering rule.
_SMALL_PERIODS = dict(ema_fast=2, ema_20=3, ema_50=4, ema_100=5, ema_200=6)


def _df(rows, symbol="TESTCO", start="2024-01-01"):
    """rows: (open, high, low, close, volume) tuples, one per day."""
    dates = pd.date_range(start, periods=len(rows))
    opens, highs, lows, closes, volumes = zip(*rows)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
        }
    )


# --- Config validation -------------------------------------------------------


def test_config_rejects_non_strictly_increasing_emas():
    """Would catch: the EMA-stack ordering check missing, letting a
    misconfigured stack (e.g. two equal periods, or out of order) silently
    reach generate_signals instead of failing at construction."""
    with pytest.raises(ValueError):
        TrendLadderStrategy(ema_fast=20, ema_20=20, ema_50=50, ema_100=100, ema_200=200)
    with pytest.raises(ValueError):
        TrendLadderStrategy(ema_fast=50, ema_20=20, ema_50=10, ema_100=100, ema_200=200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"higher_close_lookback": 0},
        {"adx_period": 0},
        {"adx_threshold": -1.0},
        {"volume_window": 0},
        {"volume_multiplier": 0.0},
        {"min_body_ratio": -0.1},
        {"min_body_ratio": 1.1},
    ],
)
def test_config_rejects_invalid_parameter(kwargs):
    """Would catch: any one of these guards being dropped or loosened,
    letting a nonsensical parameter (e.g. a negative ADX threshold) through
    to silently distort the signal logic instead of failing at construction."""
    with pytest.raises(ValueError):
        TrendLadderStrategy(**kwargs)


# --- Column / empty-input contract ------------------------------------------


def test_missing_required_column_raises():
    strat = TrendLadderStrategy(**_SMALL_PERIODS)
    df = pd.DataFrame({"symbol": ["TESTCO"], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="open"):
        strat.generate_signals(df)


def test_empty_input_returns_empty_output_with_right_columns():
    strat = TrendLadderStrategy(**_SMALL_PERIODS)
    result = strat.generate_signals(
        pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    )
    assert result.empty
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Entry/exit trigger logic ------------------------------------------------


def test_dip_and_reclaim_fires_buy_on_exact_day_and_no_unentered_sell():
    """A short, fully hand-traced uptrend: 5 flat warm-up days, a 1-day dip
    below the 20-EMA (bearish candle), then a strong bull reclaim with a
    volume spike satisfying every BUY condition at once (higher-close, bull
    candle, above the EMA stack, ADX>0, volume confirmed, solid body, and
    specifically *today* reclaiming the 20-EMA from below yesterday).

    The dip day (2024-01-07) is a bearish close below the 20-EMA, but no BUY
    precedes it, so there is no position to exit and no SELL row is emitted:
    exits are armed only by an entry.

    adx_threshold=0 and volume_multiplier=1.0 isolate the test from needing
    to hand-verify exact ADX/volume-ratio values -- only their sign/direction
    matters here, which the constructed jump makes unambiguous; the exact
    trigger *day* (not just "a signal fires somewhere") is what's asserted.

    Would catch: the 20-EMA reclaim condition missing the "yesterday was
    at/below" gate (firing on any day above the EMA, not just the reclaim
    day), an exit row emitted with no entry before it, or any of the AND-ed
    BUY conditions being wired to the wrong column.
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100, 101, 102, 103, 104, 105, 103, 110, 111, 112, 113]
    opens = [99, 100, 101, 102, 103, 104, 104, 104, 110, 111, 112]
    highs = [c + 1 for c in closes]
    lows = [o - 1 for o in opens]
    volumes = [1000] * 6 + [1000, 5000, 1000, 1000, 1000]
    rows = list(zip(opens, highs, lows, closes, volumes))

    result = strat.generate_signals(_df(rows))
    assert len(result) == 1

    buy = result.iloc[0]
    assert buy["signal_type"] == "BUY"
    assert buy["date"] == pd.Timestamp("2024-01-08")
    assert buy["price"] == pytest.approx(110.0)


def test_no_exit_rows_without_an_entry_during_an_extended_slide():
    """Regression test for a real bug this project had before: an exit that
    re-fired every single day price stayed below the 20-EMA. 10 flat-price
    warm-up days (with a steadily widening high/low range so ADX has real,
    non-tied directional movement -- see the `_compute_adx` bug test below for
    what happens when up-move and down-move tie exactly), then a crash that
    STAYS down for 3 more days, with no BUY anywhere.

    Exits are armed only by an entry, so a slide with no position open emits
    nothing at all (the "fires once, not every bearish day" half of that old
    bug is pinned by the post-BUY test right below).

    Would catch: SELL rows being emitted for days with no preceding BUY,
    bloating the signals table with exits for positions that cannot exist.
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100] * 10 + [50, 50, 50]
    opens = [99] * 10 + [100, 51, 51]
    # High +2/day, low -1/day: asymmetric so up-move and down-move never tie
    # exactly (a tie is what triggers the _compute_adx bug tested below).
    highs = [101 + 2 * i for i in range(10)] + [100, 52, 52]
    lows = [98 - i for i in range(10)] + [49, 48, 48]
    volumes = [1000] * 13
    rows = list(zip(opens, highs, lows, closes, volumes))

    result = strat.generate_signals(_df(rows))
    assert result.empty


def test_exit_fires_when_the_bearish_close_comes_a_day_after_the_ema20_cross():
    """The spec's exit is "a bearish candle closes below the 20 EMA" -- it
    does not require that candle to be the day price *crossed* the EMA.

    Same BUY as test_dip_and_reclaim_fires_buy_on_exact_day_and_no_unentered_sell
    (2024-01-08). Then 2024-01-12 gaps below the 20-EMA (3-EMA here, 104.36)
    but closes as a BULL candle (95 -> 97): correctly no exit that day. On
    2024-01-13 a bearish candle (97 -> 96) closes below the EMA (100.18):
    that is the exit. Yesterday's close was already below the EMA, so a
    "yesterday >= EMA" crossing gate never lets this SELL out, and the
    position stays open until some unrelated future cross.

    2024-01-14 is another bearish close below the EMA (97.59): it must NOT
    produce a second SELL -- one exit per entry, not a row per bearish day.

    Would catch: the SELL being gated on yesterday's close being at/above
    the 20-EMA (silently dropping every exit whose bearish candle comes the
    day after the cross), a SELL firing on the bull gap-down day itself, or
    the exit re-firing on every later bearish day below the EMA.
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100, 101, 102, 103, 104, 105, 103, 110, 111, 112, 113, 97, 96, 95]
    opens = [99, 100, 101, 102, 103, 104, 104, 104, 110, 111, 112, 95, 97, 96]
    highs = [max(o, c) + 1 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 1 for o, c in zip(opens, closes)]
    volumes = [1000] * 6 + [1000, 5000] + [1000] * 6
    rows = list(zip(opens, highs, lows, closes, volumes))

    result = strat.generate_signals(_df(rows))
    after_buy = result[result["date"] > pd.Timestamp("2024-01-08")]
    assert list(after_buy["signal_type"]) == ["SELL"]
    assert after_buy.iloc[0]["date"] == pd.Timestamp("2024-01-13")
    assert after_buy.iloc[0]["price"] == pytest.approx(96.0)


def test_no_signals_when_ema_stack_is_not_ascending():
    """A sideways/choppy series where price oscillates without ever forming
    a clean ascending EMA stack -- BUY must never fire regardless of any
    single day's momentum, since ema_stack is one of the AND-ed conditions.

    Would catch: the ema_stack condition being accidentally OR-ed instead of
    AND-ed with the rest, or omitted entirely.
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100, 102, 99, 103, 98, 104, 97, 105, 96, 106, 95, 107]
    opens = [c - 1 for c in closes]
    highs = [c + 2 for c in closes]
    lows = [c - 2 for c in closes]
    volumes = [1000] * len(closes)
    rows = list(zip(opens, highs, lows, closes, volumes))

    result = strat.generate_signals(_df(rows))
    assert result[result["signal_type"] == "BUY"].empty


def test_output_columns_and_strategy_name():
    strat = TrendLadderStrategy(**_SMALL_PERIODS)
    assert strat.name == "trend_ladder"
    result = strat.generate_signals(
        pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    )
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Regression test: _compute_adx used to crash on legitimate flat input --


def test_compute_adx_does_not_crash_on_flat_high_low_input():
    """A day (or two) with literally no net directional movement is a
    realistic input (a quiet, tightly range-bound trading day), not an edge
    case that should be excluded by construction -- _compute_adx must not
    raise on it.
    """
    high = pd.Series([10.0, 10.0, 10.0])
    low = pd.Series([5.0, 5.0, 5.0])
    close = pd.Series([7.0, 7.0, 7.0])

    result = _compute_adx(high, low, close, period=2)
    assert result.dtype.kind == "f"


# --- Nifty 50 market-regime filter (src.market_regime) -----------------------
#
# Every test above runs with no nifty_regime_* columns at all, and all still
# pass unmodified -- that itself is the "inert when absent" contract's proof.
# The tests below attach the columns explicitly to exercise the filter.


def test_nifty_regime_bearish_suppresses_an_otherwise_valid_buy():
    """Reuses test_dip_and_reclaim_fires_buy_on_exact_day_and_no_unentered_sell's exact
    scenario (BUY would otherwise fire on 2024-01-08), but flags the market
    regime bearish on that one day. No BUY must fire.

    Note this strategy's BUY trigger is a one-day crossing event (yesterday
    at/below the 20-EMA, today above it) -- since regime-blocking the exact
    reclaim day doesn't reset any persistent state, the strategy does NOT
    retroactively fire on a later day even once the regime clears; the
    crossing window has simply passed. That's an expected consequence of
    gating a crossing-event trigger, not a bug.

    Would catch: the nifty_bullish term being dropped from buy_mask, or ANDed
    in with the wrong sense (blocking bullish days instead of bearish ones).
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100, 101, 102, 103, 104, 105, 103, 110, 111, 112, 113]
    opens = [99, 100, 101, 102, 103, 104, 104, 104, 110, 111, 112]
    highs = [c + 1 for c in closes]
    lows = [o - 1 for o in opens]
    volumes = [1000] * 6 + [1000, 5000, 1000, 1000, 1000]
    rows = list(zip(opens, highs, lows, closes, volumes))

    df = _df(rows)
    df["nifty_regime_bullish"] = True
    df.loc[df["date"] == pd.Timestamp("2024-01-08"), "nifty_regime_bullish"] = False
    df["nifty_regime_breakdown"] = False

    result = strat.generate_signals(df)
    assert result[result["signal_type"] == "BUY"].empty


def test_nifty_exit_all_fires_on_flagged_day_regardless_of_ema20_condition():
    """A day flagged as a Nifty breakdown must emit a SELL for the symbol
    even though nothing about that symbol's own price action (still above
    its 20-EMA, no crossing) would otherwise trigger one -- "exit everything"
    is unconditional across the whole loaded universe on that one day.

    Would catch: the exit-all SELL being conditioned on the symbol's own
    EMA-20 state (defeating the point of "exit everything"), or the wrong
    reason string being stored.
    """
    strat = TrendLadderStrategy(
        **_SMALL_PERIODS,
        adx_period=2,
        adx_threshold=0.0,
        volume_window=2,
        volume_multiplier=1.0,
        min_body_ratio=0.0,
        higher_close_lookback=1,
    )
    closes = [100, 101, 102, 103, 104, 105, 103, 110, 111, 112, 113]
    opens = [99, 100, 101, 102, 103, 104, 104, 104, 110, 111, 112]
    highs = [c + 1 for c in closes]
    lows = [o - 1 for o in opens]
    volumes = [1000] * 6 + [1000, 5000, 1000, 1000, 1000]
    rows = list(zip(opens, highs, lows, closes, volumes))

    df = _df(rows)
    df["nifty_regime_bullish"] = True
    df["nifty_regime_breakdown"] = False
    # 2024-01-06: the first day every indicator is warmed up (ema_200's
    # min_periods=6), still just a plain up-day well above the 20-EMA, no
    # crossing -- the normal SELL condition would never fire here on its own.
    df.loc[df["date"] == pd.Timestamp("2024-01-06"), "nifty_regime_breakdown"] = True

    result = strat.generate_signals(df)
    exit_all = result[result["reason"] == "Nifty filter exit-all"]
    assert len(exit_all) == 1
    row = exit_all.iloc[0]
    assert row["date"] == pd.Timestamp("2024-01-06")
    assert row["signal_type"] == "SELL"

    # Still exactly one row for that date (no duplicate SELL from any other
    # condition also being true that day).
    assert len(result[result["date"] == pd.Timestamp("2024-01-06")]) == 1

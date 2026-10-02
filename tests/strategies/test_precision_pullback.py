"""Correctness tests for strategies/precision_pullback.py.

Unlike the other strategies in this package, this one is a genuine
multi-day state machine (_State: COUNTING_BLUE -> WAITING_FOR_RED ->
WAITING_FOR_RECOVERY -> WAITING_FOR_PULLBACK_MARK -> WAITING_FOR_CONTINUATION
-> back to COUNTING_BLUE after a BUY). Every generate_signals() test here
uses a small ema_period/blue_days_required (3/3 instead of the 50/90
defaults) and a day-by-day hand-traced price path so each state transition
can be pinned to an exact day -- every series below was walked through the
actual EWM math (span=3, adjust=False, alpha=0.5, invariant under a flat
run) before being hardcoded.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.precision_pullback import PrecisionPullbackConfig, PrecisionPullbackStrategy

_SMALL = dict(ema_period=3, blue_days_required=3)


def _df(rows, symbol="TESTCO", start="2024-01-01"):
    """rows: (open, high, low, close) tuples, one per day."""
    dates = pd.date_range(start, periods=len(rows))
    opens, highs, lows, closes = zip(*rows)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
        }
    )


# --- Config validation -------------------------------------------------------


def test_config_rejects_nonpositive_ema_period():
    with pytest.raises(ValueError):
        PrecisionPullbackStrategy(ema_period=0)


def test_config_rejects_nonpositive_blue_days_required():
    with pytest.raises(ValueError):
        PrecisionPullbackStrategy(blue_days_required=0)


# --- Column / empty-input contract ------------------------------------------


def test_missing_required_column_raises():
    strat = PrecisionPullbackStrategy(**_SMALL)
    df = pd.DataFrame({"symbol": ["TESTCO"], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="open"):
        strat.generate_signals(df)


def test_empty_input_returns_empty_output_with_right_columns():
    strat = PrecisionPullbackStrategy(**_SMALL)
    result = strat.generate_signals(
        pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close"])
    )
    assert result.empty
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Full state-machine walk-through -----------------------------------------

# Day-by-day design (ema_period=3, blue_days_required=3), hand-traced against
# the span=3/adjust=False EWM formula (alpha=0.5, invariant while flat):
#   days 0-4: flat at 100 -> EMA warms up at day 2 and stays at 100; days
#             2,3,4 are each a "blue" day (close >= EMA), reaching the
#             required 3-day streak at day 4 -> qualifies the uptrend.
#   day 5:    drops to 90 (EMA becomes 95) -- the first RED day (the
#             pullback) -- and it's a bearish candle closing below the EMA,
#             so it ALSO fires the (state-independent) SELL crossing.
#   day 6:    drops further to 88 (EMA 91.5) -- still red, tolerated,
#             no qualifying recovery candle yet.
#   day 7:    bull candle open=100/close=105, both strictly above the EMA
#             (98.25) -- the recovery candle. running_max_high starts at
#             this day's high (106).
#   day 8:    up day (close 110 > close 105) -- running_max_high updates to
#             its high (111).
#   day 9:    down day (close 108 < close 110) -- freezes mark = 111 (the
#             running max as of *before* today, i.e. day 8's high).
#   day 10:   bull candle closing at 112, above the mark (111) -- the BUY.
_FULL_ROWS = [
    (99, 101, 98, 100),
    (99, 101, 98, 100),
    (99, 101, 98, 100),
    (99, 101, 98, 100),
    (99, 101, 98, 100),
    (95, 96, 87, 90),
    (89, 90, 86, 88),
    (100, 106, 99, 105),
    (106, 111, 105, 110),
    (110, 112, 107, 108),
    (109, 113, 108, 112),
]


def test_full_pullback_cycle_fires_sell_then_buy_on_exact_days():
    """Would catch: a wrong transition condition anywhere in the five-state
    machine (e.g. the recovery check not requiring the full body above the
    band, or the mark being frozen using the wrong day's high), any of which
    would shift the BUY off day 10 or suppress it entirely.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    result = strat.generate_signals(_df(_FULL_ROWS))

    assert len(result) == 2
    sell = result[result["signal_type"] == "SELL"].iloc[0]
    assert sell["date"] == pd.Timestamp("2024-01-06")
    assert sell["price"] == pytest.approx(90.0)

    buy = result[result["signal_type"] == "BUY"].iloc[0]
    assert buy["date"] == pd.Timestamp("2024-01-11")
    assert buy["price"] == pytest.approx(112.0)


def test_pullback_with_no_qualifying_recovery_candle_never_buys():
    """Same setup truncated right after the pullback begins (day 6) -- no
    recovery candle ever appears, so the state machine must stay in
    WAITING_FOR_RECOVERY forever and never reach a BUY.

    Would catch: a BUY firing without a genuine qualifying recovery/
    continuation sequence (e.g. a state check that's too permissive).
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    result = strat.generate_signals(_df(_FULL_ROWS[:7]))
    assert result[result["signal_type"] == "BUY"].empty
    # the pullback's initial red day still fires its independent SELL crossing
    assert len(result) == 1
    assert result.iloc[0]["signal_type"] == "SELL"


def test_red_day_resets_the_blue_streak_count():
    """A red (below-EMA) day partway through the qualifying streak must
    reset the count to zero, not merely pause it -- so hitting
    blue_days_required=3 "blue" days total (with a gap) must NOT qualify.

    Would catch: the blue-streak counter not being reset to 0 on a red day
    (e.g. only decremented, or left unchanged), which would let a
    fragmented, interrupted streak qualify an uptrend that was never
    actually unbroken.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    # close: 100,100,[95 red -> resets],100,100,100 -- only 3 blue days ever
    # run consecutively (the last three), but they start fresh after the
    # reset, so by day 5 (index 5) the streak is only 3 long, not yet having
    # produced any RED/pullback event within this short window.
    rows = [
        (99, 101, 98, 100),
        (99, 101, 98, 100),
        (99, 101, 80, 95),  # red: resets the blue streak
        (99, 101, 98, 100),
        (99, 101, 98, 100),
        (99, 101, 98, 100),
    ]
    result = strat.generate_signals(_df(rows))
    assert result.empty


def test_output_columns_and_strategy_name():
    strat = PrecisionPullbackStrategy(**_SMALL)
    assert strat.name == "precision_pullback"
    result = strat.generate_signals(_df(_FULL_ROWS))
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert (result["strategy"] == strat.name).all()


def test_symbols_do_not_leak_into_each_others_state():
    """Two symbols in one input frame -- one runs the full qualifying cycle,
    the other stays flat throughout (never even starts a pullback). Only the
    first should produce any signals.

    Would catch: state (blue_streak, running_max_high, mark) being shared
    across symbols instead of reset per group, which would let one symbol's
    progress leak into another's scan.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    cycling = _df(_FULL_ROWS, symbol="CYCLES")
    flat = _df([(99, 101, 98, 100)] * len(_FULL_ROWS), symbol="FLAT")
    combined = pd.concat([cycling, flat], ignore_index=True)

    result = strat.generate_signals(combined)
    assert set(result["symbol"]) == {"CYCLES"}


# --- Nifty 50 market-regime filter (src.market_regime) -----------------------
#
# Every test above runs with no nifty_regime_* columns at all, and all still
# pass unmodified -- that itself is the "inert when absent" contract's proof.
# The tests below attach the columns explicitly to exercise the filter.


def test_nifty_regime_bearish_suppresses_the_continuation_buy():
    """Reuses test_full_pullback_cycle's exact scenario (BUY would otherwise
    fire on 2024-01-11), but flags the market regime bearish on that one day.
    No BUY must fire.

    Would catch: the nifty_bullish term being dropped from the continuation
    condition, or ANDed in with the wrong sense.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    df = _df(_FULL_ROWS)
    df["nifty_regime_bullish"] = True
    df.loc[df["date"] == pd.Timestamp("2024-01-11"), "nifty_regime_bullish"] = False
    df["nifty_regime_breakdown"] = False

    result = strat.generate_signals(df)
    assert result[result["signal_type"] == "BUY"].empty


def test_nifty_regime_bearish_defers_continuation_buy_to_a_later_bullish_day():
    """Same as above, but with one more qualifying bull day appended after
    the blocked one, where the regime has turned bullish again. Blocking a
    single day must not reset the WAITING_FOR_CONTINUATION state (the mark
    stays frozen) -- the BUY should fire on the next day the continuation
    condition holds AND the regime is bullish, not be lost entirely.

    Would catch: the regime gate accidentally resetting state (mark/streak)
    on a blocked day instead of just skipping that day's emission.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    extra_row = (112, 115, 111, 114)  # bull candle, close 114 > mark (111)
    rows = _FULL_ROWS + [extra_row]
    df = _df(rows)
    df["nifty_regime_bullish"] = True
    df.loc[df["date"] == pd.Timestamp("2024-01-11"), "nifty_regime_bullish"] = False
    df["nifty_regime_breakdown"] = False

    result = strat.generate_signals(df)
    buy = result[result["signal_type"] == "BUY"]
    assert len(buy) == 1
    assert buy.iloc[0]["date"] == pd.Timestamp("2024-01-12")
    assert buy.iloc[0]["price"] == pytest.approx(114.0)


def test_nifty_exit_all_fires_regardless_of_state():
    """A day flagged as a Nifty breakdown must emit a SELL for the symbol
    even during COUNTING_BLUE (nothing about this symbol's own price action
    would otherwise trigger anything that day) -- "exit everything" is
    unconditional across the whole loaded universe on that one day.

    Would catch: the exit-all SELL being conditioned on the entry-side
    state machine, or the wrong reason string being stored.
    """
    strat = PrecisionPullbackStrategy(**_SMALL)
    df = _df(_FULL_ROWS)
    df["nifty_regime_bullish"] = True
    df["nifty_regime_breakdown"] = False
    # 2024-01-04: still flat/warm-up (COUNTING_BLUE), nothing else would fire.
    df.loc[df["date"] == pd.Timestamp("2024-01-04"), "nifty_regime_breakdown"] = True

    result = strat.generate_signals(df)
    exit_all = result[result["reason"] == "Nifty filter exit-all"]
    assert len(exit_all) == 1
    row = exit_all.iloc[0]
    assert row["date"] == pd.Timestamp("2024-01-04")
    assert row["signal_type"] == "SELL"
    assert len(result[result["date"] == pd.Timestamp("2024-01-04")]) == 1

"""Correctness tests for strategies/illiquidity_tilt.py.

Like bollinger_reversion, this strategy is cross-sectional -- but unlike it,
entries/exits only happen on a fixed REBALANCE schedule (every
rebalance_every_days trading days), driven by a single GLOBAL `held` set
across the whole date range, not independent per-symbol cycles. Every
hand-traced test below uses a small window (2) and rebalance interval (2)
with 2-3 symbols so each rebalance day's ranking and illiquidity values can
be computed by hand; amihud illiquidity values were independently
recomputed (same formula as research/signal_library.py's
amihud_illiquidity) before being hardcoded as expectations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.illiquidity_tilt import IlliquidityTiltConfig, IlliquidityTiltStrategy

_SMALL = dict(window=2, top_quantile=0.3, rebalance_every_days=2)


def _panel(series: dict[str, dict[str, list[float]]], start="2024-01-01") -> pd.DataFrame:
    """series: {symbol: {"close": [...], "volume": [...]}}. adj_close == close
    (no corporate action) unless a symbol's dict provides its own "adj_close"."""
    length = len(next(iter(series.values()))["close"])
    dates = pd.date_range(start, periods=length)
    frames = []
    for symbol, cols in series.items():
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "close": cols["close"],
                "adj_close": cols.get("adj_close", cols["close"]),
                "volume": cols["volume"],
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --- Config validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"top_quantile": 0.0},
        {"top_quantile": 1.0},
        {"top_quantile": 1.5},
        {"rebalance_every_days": 0},
    ],
)
def test_config_rejects_invalid_parameter(kwargs):
    with pytest.raises(ValueError):
        IlliquidityTiltStrategy(**kwargs)


def test_name_folds_window_and_rebalance_cadence():
    strat = IlliquidityTiltStrategy(window=20, rebalance_every_days=63)
    assert strat.name == "illiquidity_tilt_20_63"


# --- Column / empty-input contract ------------------------------------------


def test_missing_required_column_raises():
    strat = IlliquidityTiltStrategy(**_SMALL)
    df = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-01")], "adj_close": [100.0]})
    with pytest.raises(ValueError, match="close|volume"):
        strat.generate_signals(df)


def test_empty_input_returns_empty_output_with_right_columns():
    strat = IlliquidityTiltStrategy(**_SMALL)
    result = strat.generate_signals(pd.DataFrame(columns=["symbol", "date", "adj_close", "close", "volume"]))
    assert result.empty
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Core rebalance mechanics -------------------------------------------------

# Hand-traced setup (window=2, rebalance_every_days=2, top_quantile=0.3 ->
# only the single highest-illiquidity symbol of the three ever qualifies,
# since rank_pct for 3 symbols is {0.333, 0.667, 1.0} and the threshold is
# 1 - 0.3 = 0.7): A jumps 100->150 once at day 2 (one-off level shift, not a
# spike-and-revert -- a spike-and-revert creates TWO large daily returns,
# independently verified to keep A's illiquidity elevated two extra days,
# which would confuse this specific trace), B jumps once at day 4, C
# oscillates mildly throughout and never qualifies. Independently
# recomputed via the exact amihud formula before hardcoding:
#   day index: 0     1     2       3       4       5
#   A close:  100   100   150     150     150     150
#   B close:  100   100   100     100     150     150
#   C close:  100   101   100     101     100     101
#   rank_pct:  -     -   A=1.00  A=1.00  B=1.00  B=1.00
#                        B=0.33  B=0.33  A=0.33  A=0.33
#                        C=0.67  C=0.67  C=0.67  C=0.67
# Rebalance dates (every 2 days starting at index 0): 0, 2, 4 -- index 0 has
# no valid illiquidity yet (warm-up) and produces nothing.


def _trace_panel():
    return _panel(
        {
            "A": {"close": [100, 100, 150, 150, 150, 150], "volume": [1000] * 6},
            "B": {"close": [100, 100, 100, 100, 150, 150], "volume": [1000] * 6},
            "C": {"close": [100, 101, 100, 101, 100, 101], "volume": [1000] * 6},
        }
    )


def test_rebalance_only_happens_on_the_scheduled_cadence():
    """No signals at all on day 0 (the only day before day 2, the first
    rebalance) -- confirms non-rebalance days are silently skipped, not
    just that rebalance days work."""
    strat = IlliquidityTiltStrategy(**_SMALL)
    result = strat.generate_signals(_trace_panel())
    assert result[result["date"] == pd.Timestamp("2024-01-01")].empty  # day 0


def test_first_rebalance_buys_the_single_qualifying_symbol():
    """At day 2, only A clears the 0.7 rank_pct threshold (rank 1.0); B
    (0.33) and C (0.67) do not. Would catch the threshold comparison using
    the wrong direction (bottom instead of top) or an off-by-one in which
    day counts as the first rebalance.
    """
    strat = IlliquidityTiltStrategy(**_SMALL)
    result = strat.generate_signals(_trace_panel())
    day2 = result[result["date"] == pd.Timestamp("2024-01-03")]  # index 2
    assert len(day2) == 1
    assert day2.iloc[0]["symbol"] == "A"
    assert day2.iloc[0]["signal_type"] == "BUY"
    assert day2.iloc[0]["price"] == pytest.approx(150.0)


def test_second_rebalance_sells_the_dropout_and_buys_the_new_entrant():
    """At day 4, B now has rank 1.0 (qualifies) and A has dropped to 0.33
    (no longer qualifies) -- A must be SOLD and B BOUGHT, both dated day 4,
    and C must never appear at all (it never qualifies).

    Would catch: `held` not being updated/compared correctly across
    rebalances, or a symbol that's simply never bought (C) spuriously
    appearing in a sell/buy list.
    """
    strat = IlliquidityTiltStrategy(**_SMALL)
    result = strat.generate_signals(_trace_panel())
    day4 = result[result["date"] == pd.Timestamp("2024-01-05")]  # index 4

    assert set(result["symbol"]) == {"A", "B"}  # C never trades, ever
    assert len(day4) == 2
    sell = day4[day4["signal_type"] == "SELL"].iloc[0]
    buy = day4[day4["signal_type"] == "BUY"].iloc[0]
    assert sell["symbol"] == "A"
    assert buy["symbol"] == "B"
    assert buy["price"] == pytest.approx(150.0)


# --- Missing-data handling ----------------------------------------------------


def test_held_symbol_missing_on_a_rebalance_day_is_left_untouched():
    """A is bought at day 2 (as above). If A's row is entirely ABSENT from
    the input at day 4 (a one-day data gap, not just a disqualifying rank),
    A must NOT be sold that day -- there's nothing to rank it against, so
    it's left exactly as held. B's own buy must still fire normally,
    unaffected by A's gap.

    Would catch: computing `to_sell` as `held - target` (which would
    wrongly include any held symbol missing from `valid`, since target is
    built only from `valid`) instead of `(held & valid_symbols) - target`
    -- the latter is required for this to be correct, and also for
    `prices[symbol]` to never KeyError on a missing symbol.
    """
    strat = IlliquidityTiltStrategy(**_SMALL)
    df = _trace_panel()
    # Remove A's entire row at day index 4 (2024-01-05).
    df = df[~((df["symbol"] == "A") & (df["date"] == pd.Timestamp("2024-01-05")))]

    result = strat.generate_signals(df)
    day4 = result[result["date"] == pd.Timestamp("2024-01-05")]

    assert "A" not in set(day4["symbol"])  # no SELL (or anything else) for A
    buy = day4[(day4["symbol"] == "B") & (day4["signal_type"] == "BUY")]
    assert len(buy) == 1  # B's own entry is unaffected


# --- Single-symbol input (opposite behavior from bollinger_reversion) -------


def test_single_symbol_input_always_buys_once_warmed_up():
    """Unlike BollingerReversionStrategy (bottom_quantile: a lone symbol can
    never qualify), this strategy's TOP-quantile check means a lone symbol
    always ranks at the 100th percentile, which always clears any
    top_quantile threshold -- so a single-symbol universe buys as soon as
    its illiquidity is defined. Documenting this explicit, opposite-of-the-
    sibling-strategy behavior so it's never mistaken for a bug later.
    """
    strat = IlliquidityTiltStrategy(**_SMALL)
    df = _panel({"A": {"close": [100, 100, 150, 150], "volume": [1000] * 4}})
    result = strat.generate_signals(df)
    assert (result["symbol"] == "A").all()
    assert "BUY" in set(result["signal_type"])


def test_output_columns_and_strategy_name():
    strat = IlliquidityTiltStrategy(**_SMALL)
    result = strat.generate_signals(_trace_panel())
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert (result["strategy"] == strat.name).all()

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
        {"stop_loss_pct": 0.0},
        {"stop_loss_pct": -5.0},
    ],
)
def test_config_rejects_invalid_parameter(kwargs):
    with pytest.raises(ValueError):
        IlliquidityTiltStrategy(**kwargs)


def test_stop_loss_defaults_to_12_pct_enabled():
    """Validated via a real backtest sweep (see the module docstring's 'Why
    a stop-loss' section) -- 12% improved CAGR, Sharpe, AND max drawdown
    simultaneously on both the Nifty 50 and Nifty 500, so it's the default,
    not an opt-in."""
    assert IlliquidityTiltStrategy().config.stop_loss_pct == 12.0


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


# --- Per-position stop-loss ---------------------------------------------------
#
# Setup shared by every stop-loss test below: window=2, top_quantile=0.3,
# rebalance_every_days=3 (rebalance dates 0, 3, 6, ...) with 3 symbols, same
# illiquidity-formula convention as _trace_panel -- A jumps once at day 3
# (illiquidity spikes, independently recomputed as 1.67e-6 at day 3, versus
# B/C's oscillation giving ~1e-7), so A is the single qualifier (threshold
# 0.7 for top_quantile=0.3 with 3 symbols) and gets BOUGHT at day 3 @150.
# B and C oscillate mildly throughout and never qualify, same as
# _trace_panel, so they never interfere with any assertion below. The crash
# is deliberately placed at day 5 -- a NON-rebalance day (next rebalance is
# day 6) -- so a stop-loss firing there can be asserted in isolation, with
# no same-day rebalance logic to interact with it.

_STOP_SMALL = dict(window=2, top_quantile=0.3, rebalance_every_days=3, stop_loss_pct=10.0)


def _stop_panel(a_close: list[float]) -> pd.DataFrame:
    """8-day panel; only A's close path varies between tests. B/C are fixed
    flat-oscillating non-qualifiers (see module-level comment above)."""
    bc = [100, 101, 100, 101, 100, 101, 100, 101][: len(a_close)]
    return _panel(
        {
            "A": {"close": a_close, "volume": [1000] * len(a_close)},
            "B": {"close": bc, "volume": [1000] * len(bc)},
            "C": {"close": bc, "volume": [1000] * len(bc)},
        }
    )


def test_stop_loss_fires_on_a_non_rebalance_day():
    """A is bought at day 3 (@150). Day 5 (not a rebalance day) crashes to
    130, a -13.33% move from entry -- breaches the 10% stop. The SELL must
    land on day 5, with no rebalance involved at all (day 5 isn't one).

    Would catch: the stop-loss check being skipped on non-rebalance days,
    or evaluated against the wrong reference price (e.g. the previous
    day's close instead of the position's own entry price).
    """
    strat = IlliquidityTiltStrategy(**_STOP_SMALL)
    a_close = [100, 100, 100, 150, 150, 130, 130, 130]
    result = strat.generate_signals(_stop_panel(a_close))
    a_rows = result[result["symbol"] == "A"].sort_values("date").reset_index(drop=True)

    assert a_rows.iloc[0]["signal_type"] == "BUY"
    assert a_rows.iloc[0]["date"] == pd.Timestamp("2024-01-04")  # day 3
    stop_row = a_rows[(a_rows["signal_type"] == "SELL") & (a_rows["date"] == pd.Timestamp("2024-01-06"))]  # day 5
    assert len(stop_row) == 1
    assert stop_row.iloc[0]["price"] == pytest.approx(130.0)
    assert "Stop-loss" in stop_row.iloc[0]["reason"]


def test_stop_loss_exact_boundary_is_inclusive():
    """Exactly -10% (150 -> 135) must trigger; -9.33% (150 -> 136) must not,
    leaving A held through day 5 with no SELL at all that day (day 5 is not
    a rebalance day, so nothing else could emit one).

    Would catch: an off-by-a-sign or strict-inequality mistake at the exact
    threshold.
    """
    strat = IlliquidityTiltStrategy(**_STOP_SMALL)

    exact = [100, 100, 100, 150, 150, 135, 135, 135]
    result_exact = strat.generate_signals(_stop_panel(exact))
    sells_exact = result_exact[
        (result_exact["symbol"] == "A")
        & (result_exact["signal_type"] == "SELL")
        & (result_exact["date"] == pd.Timestamp("2024-01-06"))
    ]
    assert len(sells_exact) == 1
    assert "Stop-loss" in sells_exact.iloc[0]["reason"]

    just_above = [100, 100, 100, 150, 150, 136, 136, 136]
    result_above = strat.generate_signals(_stop_panel(just_above))
    sells_above = result_above[
        (result_above["symbol"] == "A") & (result_above["signal_type"] == "SELL") & (result_above["date"] == pd.Timestamp("2024-01-06"))
    ]
    assert sells_above.empty


def test_stop_loss_can_be_explicitly_disabled():
    """Same crash as test_stop_loss_fires_on_a_non_rebalance_day, but with
    stop_loss_pct explicitly set to None -- no SELL may appear on day 5 at
    all, since that's not a rebalance day and the stop is off.

    Would catch: the stop-loss check running even when stop_loss_pct is
    None (e.g. treating None as 0 instead of "disabled").
    """
    strat = IlliquidityTiltStrategy(window=2, top_quantile=0.3, rebalance_every_days=3, stop_loss_pct=None)
    a_close = [100, 100, 100, 150, 150, 130, 130, 130]
    result = strat.generate_signals(_stop_panel(a_close))
    day5 = result[result["date"] == pd.Timestamp("2024-01-06")]
    assert day5.empty


def test_stop_loss_allows_fresh_re_entry_at_next_rebalance():
    """After the day-5 stop-loss, A's own rolling illiquidity is still
    elevated by the crash itself, so it qualifies again at the day-6
    rebalance and is bought fresh -- the stop-loss must not permanently
    lock a symbol out.

    Would catch: a stop-loss SELL leaving stale state that blocks a later,
    independently-qualifying BUY for the same symbol.
    """
    strat = IlliquidityTiltStrategy(**_STOP_SMALL)
    a_close = [100, 100, 100, 150, 150, 130, 130, 130]
    result = strat.generate_signals(_stop_panel(a_close))
    a_rows = result[result["symbol"] == "A"].sort_values("date").reset_index(drop=True)

    assert list(a_rows["signal_type"]) == ["BUY", "SELL", "BUY"]
    assert a_rows.iloc[2]["date"] == pd.Timestamp("2024-01-07")  # day 6: fresh entry allowed
    assert a_rows.iloc[2]["price"] == pytest.approx(130.0)


def test_stop_loss_missing_price_on_a_day_is_left_held_not_stopped():
    """A is held after day 3. If A's row is entirely absent at day 5 (a
    data gap, not a price), there's nothing to evaluate the stop against --
    it must be left held, not force-sold, and must not KeyError.

    Would catch: a lookup that raises or silently treats a missing day as
    a breach instead of skipping it.
    """
    strat = IlliquidityTiltStrategy(**_STOP_SMALL)
    a_close = [100, 100, 100, 150, 150, 130, 130, 130]
    df = _stop_panel(a_close)
    df = df[~((df["symbol"] == "A") & (df["date"] == pd.Timestamp("2024-01-06")))]  # remove A's day-5 row

    result = strat.generate_signals(df)
    assert result[result["date"] == pd.Timestamp("2024-01-06")].empty  # no stop fired for A

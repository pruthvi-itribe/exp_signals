"""Correctness tests for strategies/bollinger_reversion.py.

Unlike every other strategy in this package, this one is cross-sectional:
entry depends on a symbol's rank *relative to every other symbol in the same
generate_signals() call on the same date*, not on that symbol's own price
history in isolation. Every hand-traced test below uses a small window (3)
and holding period (2) with 3 symbols so each day's cross-sectional rank can
be computed by hand; bb_position values were independently recomputed
(rolling mean/std via the same formula as research/signal_library.py's
bb_position) before being hardcoded as expectations.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.bollinger_reversion import BollingerReversionConfig, BollingerReversionStrategy

_SMALL = dict(window=3, num_std=2.0, bottom_quantile=0.4, holding_period_days=2, stop_loss_pct=10.0)


def _panel(series: dict[str, list[float]], start="2024-01-01") -> pd.DataFrame:
    """series: {symbol: [adj_close, ...]} -- all series must be the same length."""
    length = len(next(iter(series.values())))
    dates = pd.date_range(start, periods=length)
    frames = []
    for symbol, closes in series.items():
        frames.append(pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes}))
    return pd.concat(frames, ignore_index=True)


# --- Config validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"num_std": 0.0},
        {"num_std": -1.0},
        {"bottom_quantile": 0.0},
        {"bottom_quantile": 1.0},
        {"bottom_quantile": 1.5},
        {"holding_period_days": 0},
        {"stop_loss_pct": 0.0},
        {"stop_loss_pct": -5.0},
    ],
)
def test_config_rejects_invalid_parameter(kwargs):
    with pytest.raises(ValueError):
        BollingerReversionStrategy(**kwargs)


def test_name_folds_window_and_holding_period():
    strat = BollingerReversionStrategy(window=30, holding_period_days=40)
    assert strat.name == "bollinger_reversion_30_40"


# --- Column / empty-input contract ------------------------------------------


def test_missing_required_column_raises():
    strat = BollingerReversionStrategy(**_SMALL)
    df = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="adj_close"):
        strat.generate_signals(df)


def test_empty_input_returns_empty_output_with_right_columns():
    strat = BollingerReversionStrategy(**_SMALL)
    result = strat.generate_signals(pd.DataFrame(columns=["symbol", "date", "adj_close"]))
    assert result.empty
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Cross-sectional entry / fixed-horizon exit ------------------------------


def test_single_symbol_input_never_buys():
    """Documented, intentional behavior: ranking one symbol against itself
    always gives it the 100th percentile, which can never fall inside
    bottom_quantile. Would catch an accidental fallback to some per-symbol
    absolute threshold that would let a lone symbol trade."""
    strat = BollingerReversionStrategy(**_SMALL)
    df = _panel({"A": [100, 100, 100, 50, 100, 100, 100]})
    result = strat.generate_signals(df)
    assert result.empty


def test_cross_sectional_entry_and_fixed_exit_on_exact_days():
    """A clearly cheapens relative to B/C on day 3 (index 3) -- bb_position
    ~0.21 for A vs ~0.65 for B/C, so A's rank_pct (0.333) clears the 0.4
    bottom_quantile cutoff while B/C's (0.833, tied) does not. Exit must
    fire exactly `holding_period_days` TRADING days later (day 5), and B/C
    must never trade at all.

    Would catch: ranking computed per-symbol instead of per-date (which
    would make every symbol "qualify" against its own history), the
    bottom_quantile comparison using the wrong sign/direction, or the exit
    offset being counted in calendar days instead of row position.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    df = _panel(
        {
            "A": [100, 100, 100, 80, 100, 100, 100],
            "B": [100, 101, 100, 101, 100, 101, 100],
            "C": [100, 101, 100, 101, 100, 101, 100],
        }
    )
    result = strat.generate_signals(df)

    assert set(result["symbol"]) == {"A"}
    assert len(result) == 2

    buy = result[result["signal_type"] == "BUY"].iloc[0]
    assert buy["date"] == pd.Timestamp("2024-01-04")  # index 3
    assert buy["price"] == pytest.approx(80.0)

    sell = result[result["signal_type"] == "SELL"].iloc[0]
    assert sell["date"] == pd.Timestamp("2024-01-06")  # index 5 (entry index 3 + holding 2)
    assert sell["price"] == pytest.approx(100.0)


def test_already_in_cycle_symbol_does_not_re_enter():
    """A stays in the bottom quantile for two consecutive days (index 3 AND
    4), but must fire only ONE buy -- the second qualifying day must be
    ignored because A is already in an active holding cycle.

    Would catch: the entry check firing every day a symbol stays in the
    bottom quantile instead of only on the day it's not already in a cycle,
    which would bloat the signals table and violate the "one entry per
    cycle" contract this strategy is built around.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    df = _panel(
        {
            "A": [100, 100, 100, 80, 78, 100, 100],
            "B": [100, 101, 100, 101, 100, 101, 100],
            "C": [100, 101, 100, 101, 100, 101, 100],
        }
    )
    result = strat.generate_signals(df)

    a_buys = result[(result["symbol"] == "A") & (result["signal_type"] == "BUY")]
    assert len(a_buys) == 1
    assert a_buys.iloc[0]["date"] == pd.Timestamp("2024-01-04")  # index 3, the FIRST qualifying day

    a_sells = result[(result["symbol"] == "A") & (result["signal_type"] == "SELL")]
    assert len(a_sells) == 1
    assert a_sells.iloc[0]["date"] == pd.Timestamp("2024-01-06")  # index 3 + holding 2 = index 5


def test_symbols_cycle_independently_at_different_times():
    """A dips and fully completes its cycle (buy day 3, sell day 5) before B
    dips later (buy day 6, sell day 8). C never dips and must never trade.
    Each symbol's cycle must be entirely self-contained.

    Would catch: cycle state (in_cycle / entry_idx) being shared across
    symbols instead of tracked independently per symbol.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    # All three share an IDENTICAL linear-drift baseline (100 + 0.1*i) --
    # a perfectly linear series has a provably constant bb_position (0.75
    # here, independent of slope), so all three are in an EXACT 3-way tie
    # on every non-dip day (verified by hand: pandas averages tied ranks,
    # giving all three pct=2/3=0.667, never clearing the 0.4 cutoff). Each
    # symbol's dip is a single day that overrides the baseline, breaking
    # the tie sharply in just that symbol's favor that day. An earlier
    # version of this test used an independent oscillating pattern per
    # symbol and kept producing accidental extra qualifications on
    # "normal" days from essentially-arbitrary near-ties -- this tied
    # baseline removes that ambiguity entirely rather than papering over it
    # with a wider margin that might not actually be wide enough.
    base = [100 + 0.1 * i for i in range(9)]
    a_values = list(base)
    a_values[3] = 80.0
    b_values = list(base)
    b_values[6] = 70.0
    df = _panel({"A": a_values, "B": b_values, "C": base})
    result = strat.generate_signals(df)

    assert set(result["symbol"]) == {"A", "B"}

    a_rows = result[result["symbol"] == "A"].sort_values("date")
    assert list(a_rows["signal_type"]) == ["BUY", "SELL"]
    assert a_rows.iloc[0]["date"] == pd.Timestamp("2024-01-04")  # index 3
    assert a_rows.iloc[0]["price"] == pytest.approx(80.0)
    assert a_rows.iloc[1]["date"] == pd.Timestamp("2024-01-06")  # index 3 + holding 2 = index 5
    assert a_rows.iloc[1]["price"] == pytest.approx(base[5])

    b_rows = result[result["symbol"] == "B"].sort_values("date")
    assert list(b_rows["signal_type"]) == ["BUY", "SELL"]
    assert b_rows.iloc[0]["date"] == pd.Timestamp("2024-01-07")  # index 6
    assert b_rows.iloc[0]["price"] == pytest.approx(70.0)
    assert b_rows.iloc[1]["date"] == pd.Timestamp("2024-01-09")  # index 6 + holding 2 = index 8
    assert b_rows.iloc[1]["price"] == pytest.approx(base[8])


def test_output_columns_and_strategy_name():
    strat = BollingerReversionStrategy(**_SMALL)
    df = _panel(
        {
            "A": [100, 100, 100, 80, 100, 100, 100],
            "B": [100, 101, 100, 101, 100, 101, 100],
            "C": [100, 101, 100, 101, 100, 101, 100],
        }
    )
    result = strat.generate_signals(df)
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert (result["strategy"] == strat.name).all()


# --- Per-position stop-loss --------------------------------------------------


def test_stop_loss_fires_before_scheduled_exit_on_breach():
    """A enters at 80 (day 3), then drops to 70 on day 4 -- a -12.5% move,
    breaching the 10% stop-loss before the scheduled exit (day 3 + holding
    2 = day 5) would otherwise fire. The SELL must land on day 4, not day 5.

    Would catch: the stop-loss check being skipped entirely, or checked
    against the wrong reference price (e.g. yesterday's close instead of
    the cycle's own entry price).
    """
    strat = BollingerReversionStrategy(**_SMALL)
    # Truncated to 5 days (indices 0-4): A's crash still partially
    # contaminates its own 3-day rolling window for a couple of days after
    # the stop-loss fires (verified by hand -- A's bb_position doesn't fully
    # decouple from the crash until its window rolls entirely past it),
    # which can trigger an unrelated fresh re-entry a day or two later.
    # Irrelevant to what this test checks, so there's no later data for it
    # to happen with.
    base = [100 + 0.1 * i for i in range(5)]
    a_values = list(base)
    a_values[3] = 80.0
    a_values[4] = 70.0  # (70-80)/80 = -12.5%, breaches the 10% stop
    df = _panel({"A": a_values, "B": base, "C": base})

    result = strat.generate_signals(df)
    a_rows = result[result["symbol"] == "A"].sort_values("date").reset_index(drop=True)

    assert list(a_rows["signal_type"]) == ["BUY", "SELL"]
    assert a_rows.iloc[1]["date"] == pd.Timestamp("2024-01-05")  # day 4, NOT day 5
    assert a_rows.iloc[1]["price"] == pytest.approx(70.0)
    assert "Stop-loss" in a_rows.iloc[1]["reason"]


def test_stop_loss_exact_boundary_is_inclusive():
    """Exactly -10% must trigger (the check is <=, not <); -9% must not,
    letting the position continue to its scheduled exit instead.

    Would catch: an off-by-a-sign or strict-inequality mistake at the exact
    threshold.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    base = [100 + 0.1 * i for i in range(7)]

    exact = list(base)
    exact[3] = 100.0
    exact[4] = 90.0  # exactly -10%
    df_exact = _panel({"A": exact, "B": base, "C": base})
    result_exact = strat.generate_signals(df_exact)
    a_exact = result_exact[result_exact["symbol"] == "A"].sort_values("date").reset_index(drop=True)
    assert a_exact.iloc[1]["date"] == pd.Timestamp("2024-01-05")  # day 4: stopped out
    assert "Stop-loss" in a_exact.iloc[1]["reason"]

    just_above = list(base)
    just_above[3] = 100.0
    just_above[4] = 91.0  # -9%, must NOT trigger
    df_above = _panel({"A": just_above, "B": base, "C": base})
    result_above = strat.generate_signals(df_above)
    a_above = result_above[result_above["symbol"] == "A"].sort_values("date").reset_index(drop=True)
    assert a_above.iloc[1]["date"] == pd.Timestamp("2024-01-06")  # day 5: scheduled exit, not stopped early
    assert "Stop-loss" not in a_above.iloc[1]["reason"]


def test_stop_loss_resets_cycle_allowing_fresh_entry():
    """After a stop-loss exit, the symbol must be eligible for a fresh
    entry on a later qualifying day, not permanently locked out.

    Would catch: the stop-loss SELL not resetting in_cycle/entry_idx/entry_price.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    base = [100 + 0.1 * i for i in range(8)]
    a_values = list(base)
    a_values[3] = 80.0  # entry 1
    a_values[4] = 70.0  # stop-loss fires here (day 4)
    a_values[6] = 50.0  # a second, fresh dip -- entry 2
    df = _panel({"A": a_values, "B": base, "C": base})

    result = strat.generate_signals(df)
    a_rows = result[result["symbol"] == "A"].sort_values("date").reset_index(drop=True)

    assert list(a_rows["signal_type"]) == ["BUY", "SELL", "BUY"]
    assert a_rows.iloc[2]["date"] == pd.Timestamp("2024-01-07")  # day 6: fresh entry allowed
    assert a_rows.iloc[2]["price"] == pytest.approx(50.0)


def test_stop_loss_coinciding_with_scheduled_exit_emits_only_one_sell():
    """If the stop-loss breach happens to land on the exact day the fixed
    holding period would also end, exactly one SELL must be emitted (the
    stop-loss), never two.

    Would catch: checking the scheduled-exit condition without excluding
    the stop-loss condition, double-emitting a SELL on the same date.
    """
    strat = BollingerReversionStrategy(**_SMALL)
    base = [100 + 0.1 * i for i in range(7)]
    a_values = list(base)
    a_values[3] = 80.0  # entry; scheduled exit = day 3 + holding 2 = day 5
    a_values[5] = 60.0  # -25% from entry, breaches stop-loss on the SAME day as the scheduled exit
    df = _panel({"A": a_values, "B": base, "C": base})

    result = strat.generate_signals(df)
    a_sells = result[(result["symbol"] == "A") & (result["signal_type"] == "SELL")]
    assert len(a_sells) == 1
    assert "Stop-loss" in a_sells.iloc[0]["reason"]

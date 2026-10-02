"""Correctness tests for strategies/bollinger_breakout.py.

Every generate_signals() test uses a small, fully hand-traceable price
series (window=3, num_std=1 unless noted) so the exact band values and the
resulting crossing days can be verified by hand -- not just eyeballed for
plausibility. See each test's docstring for the specific bug it would catch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.bollinger_breakout import BollingerBreakoutConfig, BollingerBreakoutStrategy


def _df(closes, symbol="TESTCO", start="2024-01-01"):
    dates = pd.date_range(start, periods=len(closes))
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


# --- Config validation ------------------------------------------------------


def test_config_rejects_nonpositive_window():
    """Would catch: a window of 0 or negative silently reaching the rolling
    computation instead of failing fast at construction."""
    with pytest.raises(ValueError):
        BollingerBreakoutStrategy(window=0)


def test_config_rejects_nonpositive_num_std():
    """Would catch: num_std <= 0 collapsing the bands onto the middle band
    (or inverting them) without any error."""
    with pytest.raises(ValueError):
        BollingerBreakoutStrategy(num_std=0.0)
    with pytest.raises(ValueError):
        BollingerBreakoutStrategy(num_std=-1.0)


def test_config_rejects_unsupported_exit_mode():
    """Would catch: an unrecognized exit_mode being silently accepted at
    construction and only failing later (or not at all) inside generate_signals."""
    with pytest.raises(ValueError):
        BollingerBreakoutStrategy(exit_mode="not_a_real_mode")


# --- Column / empty-input contract ------------------------------------------


def test_missing_required_column_raises():
    """Would catch: validate_columns silently skipping, or generate_signals
    reading a column that isn't guaranteed to exist."""
    strat = BollingerBreakoutStrategy()
    df = pd.DataFrame({"symbol": ["TESTCO"], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ValueError, match="adj_close"):
        strat.generate_signals(df)


def test_empty_input_returns_empty_output_with_right_columns():
    """Would catch: an empty DataFrame reaching the groupby/rolling logic and
    raising, instead of the documented empty-in/empty-out short circuit."""
    strat = BollingerBreakoutStrategy()
    result = strat.generate_signals(pd.DataFrame(columns=["symbol", "date", "adj_close"]))
    assert result.empty
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)


# --- Breakout / boundary logic ----------------------------------------------


def test_price_exactly_at_upper_band_is_not_a_breakout():
    """window=3, num_std=1 on closes [8, 10, 12, 14]: the day-3 rolling
    window is (10, 12, 14) -> mean=12, std=2 -> bb_upper=14 exactly, and the
    close (14) sits exactly ON the band, not above it.

    Would catch: a `>=` used where the code means `>` (or vice versa) in the
    breakout comparison, which would wrongly fire on a mere touch.
    """
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    result = strat.generate_signals(_df([8, 10, 12, 14]))
    assert result.empty


def test_breakout_above_upper_band_fires_buy_on_the_crossing_day_only():
    """window=3, num_std=1 on closes [8, 10, 12, 11, 50]: day 4's close (50)
    is far above its band (upper band computed from the window (12, 11, 50)
    is well under 50 regardless of the exact std), while day 3's close (11)
    stays under its own band (12) -- so day 3 must NOT fire, only day 4.

    Would catch: an off-by-one date (signal attributed to the day before/
    after the actual crossing), or a breakout firing on a day that didn't
    actually cross.
    """
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    result = strat.generate_signals(_df([8, 10, 12, 11, 50]))
    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "BUY"
    assert row["date"] == pd.Timestamp("2024-01-05")
    assert row["price"] == pytest.approx(50.0)


def test_middle_band_revert_fires_sell_once_on_the_crossing_day():
    """window=3, num_std=1 on closes [10, 10, 10, 10, 10, 4]: flat at the
    band's middle for 5 days (mean=10, std=0, so bb_middle=10 exactly, and
    close==middle counts as "at/above" for the entry side of the crossing
    check), then a sharp drop to 4, well under the day-6 middle band.

    Would catch: the exit crossing firing on every day price is below the
    middle (rather than only the crossing day), or firing on the flat days
    where close == middle.
    """
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    result = strat.generate_signals(_df([10, 10, 10, 10, 10, 4]))
    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "SELL"
    assert row["date"] == pd.Timestamp("2024-01-06")
    assert row["price"] == pytest.approx(4.0)


def test_flat_series_never_breaks_out_or_reverts():
    """A perfectly flat series has zero std, so upper == middle == lower ==
    price every day -- no crossing is possible in either direction.

    Would catch: a `>=`/`<=` boundary bug that treats "touching" the band as
    a crossing when std is 0 (a real, not just theoretical, case for a
    tightly range-bound stock).
    """
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    result = strat.generate_signals(_df([10] * 10))
    assert result.empty


def test_insufficient_history_produces_no_signals():
    """Fewer rows than `window` -- the rolling mean/std never resolve to a
    non-NaN value, so there's nothing to cross.

    Would catch: min_periods not actually being enforced (a partial-window
    band being computed and treated as real), which would produce spurious
    early signals before the indicator is properly warmed up.
    """
    strat = BollingerBreakoutStrategy(window=20, num_std=2.0)
    result = strat.generate_signals(_df([10, 20, 5, 30, 15]))
    assert result.empty


def test_symbols_do_not_leak_into_each_others_signals():
    """Two symbols in one input frame: one has a clean breakout, the other
    is flat throughout. Only the breaking-out symbol should produce a signal.

    Would catch: a per-symbol rolling computation accidentally computed over
    the whole concatenated frame (or a groupby that doesn't reset per group),
    which would let one symbol's price history leak into another's bands.
    """
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    breaking = _df([8, 10, 12, 11, 50], symbol="BREAKS")
    flat = _df([10, 10, 10, 10, 10], symbol="FLAT")
    combined = pd.concat([breaking, flat], ignore_index=True)

    result = strat.generate_signals(combined)
    assert set(result["symbol"]) == {"BREAKS"}
    assert len(result) == 1


def test_output_columns_and_strategy_name():
    """Would catch: a strategy label mismatch (e.g. hardcoded instead of
    self.name) or extra/missing/reordered output columns breaking downstream
    storage into the `signals` table."""
    strat = BollingerBreakoutStrategy(window=3, num_std=1.0)
    result = strat.generate_signals(_df([8, 10, 12, 11, 50]))
    assert list(result.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert (result["strategy"] == strat.name).all()
    assert strat.name == "bollinger_breakout"

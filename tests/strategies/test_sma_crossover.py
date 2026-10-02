"""Correctness tests for strategies/sma_crossover.py against a hand-traced
crossover sequence. fast_window=2/slow_window=3 (rather than the 20/50
defaults) are used throughout so the exact SMA values and crossover days can
be worked out by hand in a short series -- this exercises the identical
generate_signals code path as any other window pair. Each test's docstring
says what specific bug it would catch if it failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.sma_crossover import SmaCrossoverConfig, SmaCrossoverStrategy


def _df(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


# Hand-traced series (see module docstring / worksheet below for the full
# by-hand SMA table): with fast=2, slow=3 this produces exactly one BUY (at
# the row priced 20.0) and one SELL (at the row priced 10.0).
#
# idx: price  sma2    sma3
#  0:  10     NaN     NaN
#  1:  10     10.0    NaN
#  2:  10     10.0    10.000
#  3:  20     15.0    13.333   <- fast crosses above slow here: BUY
#  4:  20     20.0    16.667
#  5:  20     20.0    20.000
#  6:  10     15.0    16.667   <- fast crosses below slow here: SELL
#  7:  10     10.0    13.333
#  8:  10     10.0    10.000
#  9:  10     10.0    10.000
_TRACED_CLOSES = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 10.0]


def test_config_rejects_fast_window_not_less_than_slow_window():
    """Would catch: the fast>=slow guard being dropped or inverted, which
    would let a nonsensical crossover pair (or a crossover that never fires
    because slow reacts faster than fast) through silently.
    """
    with pytest.raises(ValueError, match="fast_window"):
        SmaCrossoverConfig(fast_window=20, slow_window=20).validate()
    with pytest.raises(ValueError, match="fast_window"):
        SmaCrossoverConfig(fast_window=50, slow_window=20).validate()
    SmaCrossoverConfig(fast_window=19, slow_window=20).validate()  # must not raise


def test_config_rejects_non_positive_windows():
    with pytest.raises(ValueError, match="positive"):
        SmaCrossoverConfig(fast_window=0, slow_window=20).validate()
    with pytest.raises(ValueError, match="positive"):
        SmaCrossoverConfig(fast_window=5, slow_window=-1).validate()


def test_invalid_config_raises_at_construction_not_at_generate_signals():
    """Per Strategy.__init__'s documented contract, validate() runs at
    construction -- would catch validation being deferred until
    generate_signals is called, which could let a broken strategy sit in a
    pipeline unnoticed until it's actually run against data.
    """
    with pytest.raises(ValueError):
        SmaCrossoverStrategy(fast_window=50, slow_window=20)


def test_name_folds_windows_into_strategy_identity():
    """Would catch: two different window configs colliding under the same
    `signals.strategy` value (breaking the table's (symbol, date, strategy)
    primary key across a parameter sweep).
    """
    assert SmaCrossoverStrategy(fast_window=10, slow_window=30).name == "sma_crossover_10_30"
    assert SmaCrossoverStrategy().name == "sma_crossover_20_50"  # documented default


def test_generate_signals_matches_hand_traced_crossovers():
    """The single BUY and single SELL from the hand-traced worksheet above,
    exactly -- not just "some signal appeared somewhere".

    Would catch: an off-by-one in the crossover comparison (e.g. comparing
    same-day values instead of prev-vs-current), a >= vs > mixup that fires
    a day early/late, or emitting a HOLD-like row on every day instead of
    only crossing events.
    """
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = _df("TESTCO", _TRACED_CLOSES)
    signals = strategy.generate_signals(df)

    assert list(signals.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert len(signals) == 2

    buy = signals.loc[signals["signal_type"] == "BUY"].iloc[0]
    sell = signals.loc[signals["signal_type"] == "SELL"].iloc[0]

    assert buy["date"] == pd.Timestamp("2024-01-04")  # idx3
    assert buy["price"] == pytest.approx(20.0)
    assert "crossed above" in buy["reason"]

    assert sell["date"] == pd.Timestamp("2024-01-07")  # idx6
    assert sell["price"] == pytest.approx(10.0)
    assert "crossed below" in sell["reason"]

    assert (signals["strategy"] == strategy.name).all()
    assert (signals["symbol"] == "TESTCO").all()


def test_flat_series_produces_no_signals():
    """A perfectly flat price never crosses anything -- fast == slow for
    every warmed row.

    Would catch: a `<=`/`>=` boundary condition that misfires a spurious
    crossover when fast and slow are exactly equal instead of requiring a
    strict cross.
    """
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = _df("FLATCO", [100.0] * 10)
    signals = strategy.generate_signals(df)
    assert signals.empty


def test_insufficient_history_produces_no_signals_not_a_crash():
    """Fewer rows than slow_window means the slow SMA never warms up.

    Would catch: an unguarded index/shift operation crashing on too-short
    input instead of returning an empty signal set.
    """
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = _df("SHORTCO", [10.0, 20.0])  # only 2 rows, slow_window needs 3
    signals = strategy.generate_signals(df)
    assert signals.empty


def test_empty_input_returns_empty_with_correct_columns():
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = pd.DataFrame(columns=["symbol", "date", "adj_close"])
    signals = strategy.generate_signals(df)
    assert signals.empty
    assert list(signals.columns) == list(SIGNAL_OUTPUT_COLUMNS)


def test_missing_required_column_raises():
    """Would catch: generate_signals skipping its own validate_columns call
    and instead failing with a confusing KeyError deep inside pandas.
    """
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"]})  # no adj_close
    with pytest.raises(ValueError, match="adj_close"):
        strategy.generate_signals(df)


def test_multi_symbol_signals_are_independent_per_symbol():
    """Two symbols, each individually equal to the hand-traced series above
    -- both must independently produce the same BUY/SELL pair, proving the
    per-symbol groupby doesn't let one symbol's rolling window leak into or
    interfere with another's.

    Would catch: a crossover comparison accidentally computed against the
    wrong symbol's previous-day values at the seam between two symbols'
    blocks of rows.
    """
    strategy = SmaCrossoverStrategy(fast_window=2, slow_window=3)
    df = pd.concat([_df("AAA", _TRACED_CLOSES), _df("BBB", _TRACED_CLOSES)], ignore_index=True)
    signals = strategy.generate_signals(df)

    assert len(signals) == 4  # 2 signals x 2 symbols
    for symbol in ("AAA", "BBB"):
        symbol_signals = signals.loc[signals["symbol"] == symbol].sort_values("date")
        assert list(symbol_signals["signal_type"]) == ["BUY", "SELL"]
        assert symbol_signals["date"].iloc[0] == pd.Timestamp("2024-01-04")
        assert symbol_signals["date"].iloc[1] == pd.Timestamp("2024-01-07")

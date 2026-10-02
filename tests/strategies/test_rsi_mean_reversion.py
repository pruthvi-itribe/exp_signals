"""Correctness tests for strategies/rsi_mean_reversion.py against a
hand-traced RSI crossing sequence. rsi_period=2 (rather than the default 14)
is used so the exact RSI values and crossing days can be worked out by hand
in a short series -- this exercises the identical generate_signals code path
as any other period. Each test's docstring says what specific bug it would
catch if it failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.base import SIGNAL_OUTPUT_COLUMNS
from strategies.rsi_mean_reversion import RsiMeanReversionConfig, RsiMeanReversionStrategy


def _df(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


# Hand-traced series (rsi_period=2; see worksheet below), oversold=30, exit=50:
#
# idx: price  change  gain  loss  avg_gain avg_loss   rs      rsi
#  0:  100     --      --    --      --       --       --      --
#  1:  105     +5      5     0       --       --       --      --   (window needs 2 valid diffs)
#  2:   95    -10      0    10      2.5       5.0      0.5    33.333
#  3:   90     -5      0     5      0.0       7.5      0.0     0.000   <- crosses BELOW 30: BUY
#  4:  100    +10     10     0      5.0       2.5      2.0    66.667   <- crosses ABOVE 50: SELL
#  5:  110    +10     10     0     10.0       0.0       --   100.000  (avg_loss==0 guard)
#  6:  115     +5      5     0      7.5       0.0       --   100.000
#  7:   90    -25      0    25      2.5      12.5      0.2    16.667   <- crosses BELOW 30: BUY
_TRACED_CLOSES = [100.0, 105.0, 95.0, 90.0, 100.0, 110.0, 115.0, 90.0]


def test_config_rejects_non_positive_rsi_period():
    with pytest.raises(ValueError, match="rsi_period"):
        RsiMeanReversionConfig(rsi_period=0).validate()


def test_config_rejects_threshold_ordering_violations():
    """Would catch: the 0 <= oversold < exit <= 100 guard being dropped,
    which would let a nonsensical (or inverted) threshold pair through --
    e.g. an "oversold" level above the "exit" level, which would make BUY
    and SELL fire on backwards conditions.
    """
    with pytest.raises(ValueError, match="oversold_threshold"):
        RsiMeanReversionConfig(oversold_threshold=50, exit_threshold=30).validate()
    with pytest.raises(ValueError, match="oversold_threshold"):
        RsiMeanReversionConfig(oversold_threshold=30, exit_threshold=30).validate()  # equal not allowed
    with pytest.raises(ValueError, match="oversold_threshold"):
        RsiMeanReversionConfig(oversold_threshold=-1, exit_threshold=50).validate()
    with pytest.raises(ValueError, match="oversold_threshold"):
        RsiMeanReversionConfig(oversold_threshold=30, exit_threshold=101).validate()
    RsiMeanReversionConfig(oversold_threshold=30, exit_threshold=50).validate()  # must not raise


def test_invalid_config_raises_at_construction():
    with pytest.raises(ValueError):
        RsiMeanReversionStrategy(oversold_threshold=60, exit_threshold=40)


def test_generate_signals_matches_hand_traced_crossings():
    """Two BUYs (entering oversold) and one SELL (exiting back above 50) in
    between them, exactly as hand-traced above -- including a BUY -> SELL ->
    BUY re-entry sequence, not just a single isolated event.

    Would catch: an off-by-one in the crossing comparison, the buy/sell
    conditions swapped, or the zero-avg_loss guard (RSI forced to 100)
    accidentally triggering a spurious SELL.
    """
    strategy = RsiMeanReversionStrategy(rsi_period=2, oversold_threshold=30, exit_threshold=50)
    df = _df("TESTCO", _TRACED_CLOSES)
    signals = strategy.generate_signals(df).sort_values("date").reset_index(drop=True)

    assert list(signals.columns) == list(SIGNAL_OUTPUT_COLUMNS)
    assert list(signals["signal_type"]) == ["BUY", "SELL", "BUY"]
    assert list(signals["date"]) == [
        pd.Timestamp("2024-01-04"),  # idx3, RSI 33.333 -> 0.000
        pd.Timestamp("2024-01-05"),  # idx4, RSI 0.000 -> 66.667
        pd.Timestamp("2024-01-08"),  # idx7, RSI 100.000 -> 16.667
    ]
    assert list(signals["price"]) == [pytest.approx(90.0), pytest.approx(100.0), pytest.approx(90.0)]
    assert (signals["strategy"] == strategy.name).all()
    assert "oversold entry" in signals["reason"].iloc[0]
    assert "mean reversion exit" in signals["reason"].iloc[1]


def test_flat_series_produces_no_signals():
    """A flat price is defined as RSI=100 for every warmed row (per the
    avg_loss==0 guard) -- constant 100 never crosses below 30 or from
    below up through 50, so no signal should ever fire.

    Would catch: the flat-market RSI=100 guard interacting badly with the
    crossing masks and spuriously emitting a SELL on the first warmed row
    (since 100 > exit_threshold could look like "just crossed" if the
    shift/mask logic were wrong).
    """
    strategy = RsiMeanReversionStrategy(rsi_period=2, oversold_threshold=30, exit_threshold=50)
    df = _df("FLATCO", [100.0] * 8)
    signals = strategy.generate_signals(df)
    assert signals.empty


def test_insufficient_history_produces_no_signals_not_a_crash():
    strategy = RsiMeanReversionStrategy(rsi_period=2, oversold_threshold=30, exit_threshold=50)
    df = _df("SHORTCO", [100.0])  # 1 row: not even one diff exists
    signals = strategy.generate_signals(df)
    assert signals.empty


def test_empty_input_returns_empty_with_correct_columns():
    strategy = RsiMeanReversionStrategy()
    df = pd.DataFrame(columns=["symbol", "date", "adj_close"])
    signals = strategy.generate_signals(df)
    assert signals.empty
    assert list(signals.columns) == list(SIGNAL_OUTPUT_COLUMNS)


def test_missing_required_column_raises():
    strategy = RsiMeanReversionStrategy()
    df = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"]})
    with pytest.raises(ValueError, match="adj_close"):
        strategy.generate_signals(df)


def test_multi_symbol_signals_are_independent_per_symbol():
    """Would catch: RSI computed across symbol boundaries (e.g. a rolling
    window bleeding one symbol's price history into another's at the seam).
    """
    strategy = RsiMeanReversionStrategy(rsi_period=2, oversold_threshold=30, exit_threshold=50)
    df = pd.concat([_df("AAA", _TRACED_CLOSES), _df("BBB", _TRACED_CLOSES)], ignore_index=True)
    signals = strategy.generate_signals(df)

    assert len(signals) == 6  # 3 signals x 2 symbols
    for symbol in ("AAA", "BBB"):
        symbol_signals = signals.loc[signals["symbol"] == symbol].sort_values("date")
        assert list(symbol_signals["signal_type"]) == ["BUY", "SELL", "BUY"]

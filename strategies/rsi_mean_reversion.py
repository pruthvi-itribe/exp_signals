"""RSI mean-reversion strategy: buy oversold dips, sell on recovery."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy


@dataclass(frozen=True)
class RsiMeanReversionConfig(StrategyConfig):
    """Tunable parameters for :class:`RsiMeanReversionStrategy`.

    Defaults reproduce the standard 14-period RSI mean-reversion setup: buy
    when RSI drops below 30 (oversold), sell when it recovers past 50.
    """

    rsi_period: int = field(
        default=14,
        metadata={"description": "Lookback period (trading days) for the RSI calculation."},
    )
    oversold_threshold: int = field(
        default=30,
        metadata={"description": "RSI level below which the market is considered oversold; crossing below it triggers a BUY."},
    )
    exit_threshold: int = field(
        default=50,
        metadata={"description": "RSI level above which momentum is considered to have faded; crossing above it triggers a SELL."},
    )

    def validate(self) -> None:
        if self.rsi_period < 1:
            raise ValueError("rsi_period must be positive.")
        if not (0 <= self.oversold_threshold < self.exit_threshold <= 100):
            raise ValueError(
                f"Require 0 <= oversold_threshold < exit_threshold <= 100; "
                f"got oversold_threshold={self.oversold_threshold}, exit_threshold={self.exit_threshold}."
            )


def _compute_rsi(adj_close: pd.Series, period: int) -> pd.Series:
    """Compute RSI via rolling average gain/loss, parameterized by ``period``.

    Mirrors ``indicators._compute_rsi_14`` exactly, with the hardcoded
    window of 14 replaced by ``period`` — so at ``period=14`` this
    reproduces the same values as the pre-computed ``rsi_14`` column in
    ``indicators_daily``. Computed here rather than read from that table
    because ``indicators_daily`` only holds one fixed period and can't serve
    a parameter sweep over arbitrary periods (see ``validate_strategy.py``),
    the same reason ``SmaCrossoverStrategy`` computes its own SMAs instead
    of reading ``sma_20``/``sma_50``.
    """
    change = adj_close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna())
    return rsi


@register_strategy("rsi_mean_reversion")
class RsiMeanReversionStrategy(Strategy):
    """RSI mean-reversion: buy oversold dips, sell on recovery past the exit level.

    Emits a BUY when RSI crosses from at/above ``oversold_threshold`` down
    below it (entering oversold territory) and a SELL when RSI crosses from
    at/below ``exit_threshold`` up above it (mean-reverting back out). Only
    crossing *events* are emitted — no HOLD rows.
    """

    base_name = "rsi_mean_reversion"
    config_cls = RsiMeanReversionConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    config: RsiMeanReversionConfig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect per-symbol RSI oversold-entry / mean-reversion-exit crossings.

        Args:
            df: Merged market data with at least ``symbol``, ``date``,
                ``adj_close`` for one or more symbols. Should include full
                per-symbol history (not just the window of interest) so RSI
                is properly warmed up by the time signals are filtered to
                whatever range the caller actually wants.

        Returns:
            Signal rows for crossing days only, matching ``SIGNAL_OUTPUT_COLUMNS``.
        """
        if df.empty:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        self.validate_columns(df)

        working = df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
        working = working.sort_values(["symbol", "date"]).reset_index(drop=True)

        signal_frames: list[pd.DataFrame] = []
        for _, group in working.groupby("symbol", sort=True):
            symbol_signals = self._generate_symbol_signals(group)
            if not symbol_signals.empty:
                signal_frames.append(symbol_signals)

        if not signal_frames:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.concat(signal_frames, ignore_index=True).loc[:, SIGNAL_OUTPUT_COLUMNS]

    def _generate_symbol_signals(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute this symbol's RSI and detect oversold-entry/exit crossings."""
        oversold_threshold = self.config.oversold_threshold
        exit_threshold = self.config.exit_threshold

        data = group.copy()
        data["rsi"] = _compute_rsi(data["adj_close"], self.config.rsi_period)
        data = data.loc[data["rsi"].notna()]
        if len(data) < 2:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        prev_rsi = data["rsi"].shift(1)
        valid_event = prev_rsi.notna()

        buy_mask = valid_event & (prev_rsi >= oversold_threshold) & (data["rsi"] < oversold_threshold)
        sell_mask = valid_event & (prev_rsi <= exit_threshold) & (data["rsi"] > exit_threshold)

        rows: list[dict[str, object]] = []
        for idx in data.index[buy_mask]:
            row = data.loc[idx]
            rows.append(
                {
                    "symbol": row["symbol"],
                    "date": row["date"],
                    "strategy": self.name,
                    "signal_type": "BUY",
                    "price": float(row["adj_close"]),
                    "reason": f"RSI crossed below {oversold_threshold} (oversold entry)",
                }
            )

        for idx in data.index[sell_mask]:
            row = data.loc[idx]
            rows.append(
                {
                    "symbol": row["symbol"],
                    "date": row["date"],
                    "strategy": self.name,
                    "signal_type": "SELL",
                    "price": float(row["adj_close"]),
                    "reason": f"RSI crossed above {exit_threshold} (mean reversion exit)",
                }
            )

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.DataFrame(rows)

"""SMA fast/slow crossover strategy."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy


@dataclass(frozen=True)
class SmaCrossoverConfig(StrategyConfig):
    """Tunable windows for :class:`SmaCrossoverStrategy`.

    Defaults (20, 50) reproduce the strategy's original fixed-window
    behavior exactly.
    """

    fast_window: int = field(
        default=20,
        metadata={"description": "Lookback window (trading days) for the fast moving average."},
    )
    slow_window: int = field(
        default=50,
        metadata={"description": "Lookback window (trading days) for the slow moving average."},
    )

    def validate(self) -> None:
        if self.fast_window < 1 or self.slow_window < 1:
            raise ValueError("fast_window and slow_window must both be positive.")
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be less than slow_window ({self.slow_window})."
            )


@register_strategy("sma_crossover")
class SmaCrossoverStrategy(Strategy):
    """SMA fast/slow crossover on ``adj_close``.

    Emits a BUY when the fast SMA crosses above the slow SMA and a SELL when
    it crosses below. Only crossover *events* are emitted — no HOLD rows.

    SMAs are computed here, directly from ``adj_close``, at whatever windows
    ``config`` specifies — not read from ``indicators_daily``, which holds
    only one fixed (20, 50) parameterization and can't serve a parameter
    sweep over arbitrary windows (see ``validate_strategy.py``).
    """

    base_name = "sma_crossover"
    config_cls = SmaCrossoverConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    config: SmaCrossoverConfig

    @property
    def name(self) -> str:
        """E.g. ``sma_crossover_20_50`` for the default config."""
        return f"{self.base_name}_{self.config.fast_window}_{self.config.slow_window}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect per-symbol fast/slow SMA crossover events.

        Args:
            df: Merged market data with at least ``symbol``, ``date``,
                ``adj_close`` for one or more symbols. Should include full
                per-symbol history (not just the window of interest) so the
                SMAs are properly warmed up by the time signals are filtered
                to whatever range the caller actually wants.

        Returns:
            Signal rows for crossover days only, matching ``SIGNAL_OUTPUT_COLUMNS``.
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
        """Compute this symbol's fast/slow SMAs and detect crossover events."""
        fast_window = self.config.fast_window
        slow_window = self.config.slow_window

        price = group["adj_close"]
        data = group.copy()
        data["sma_fast"] = price.rolling(window=fast_window, min_periods=fast_window).mean()
        data["sma_slow"] = price.rolling(window=slow_window, min_periods=slow_window).mean()

        valid = data["sma_fast"].notna() & data["sma_slow"].notna()
        data = data.loc[valid]
        if len(data) < 2:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        prev_fast = data["sma_fast"].shift(1)
        prev_slow = data["sma_slow"].shift(1)
        valid_event = prev_fast.notna() & prev_slow.notna()

        buy_mask = valid_event & (prev_fast <= prev_slow) & (data["sma_fast"] > data["sma_slow"])
        sell_mask = valid_event & (prev_fast >= prev_slow) & (data["sma_fast"] < data["sma_slow"])

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
                    "reason": (
                        f"SMA{fast_window} crossed above SMA{slow_window} "
                        f"({fast_window}d: {row['sma_fast']:.1f}, {slow_window}d: {row['sma_slow']:.1f})"
                    ),
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
                    "reason": (
                        f"SMA{fast_window} crossed below SMA{slow_window} "
                        f"({fast_window}d: {row['sma_fast']:.1f}, {slow_window}d: {row['sma_slow']:.1f})"
                    ),
                }
            )

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.DataFrame(rows)

"""Bollinger Band breakout strategy: buy upper-band breakouts, exit on mean reversion."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy

MIDDLE_BAND_REVERT: str = "middle_band_revert"
SUPPORTED_EXIT_MODES: frozenset[str] = frozenset({MIDDLE_BAND_REVERT})


@dataclass(frozen=True)
class BollingerBreakoutConfig(StrategyConfig):
    """Tunable parameters for :class:`BollingerBreakoutStrategy`.

    Defaults reproduce the standard 20-day, 2-std Bollinger Band setup —
    matching ``indicators_daily``'s ``bb_middle``/``bb_upper``/``bb_lower``
    at those defaults.
    """

    window: int = field(
        default=20,
        metadata={"description": "Lookback window (trading days) for the moving average and rolling std the bands are built from."},
    )
    num_std: float = field(
        default=2.0,
        metadata={"description": "Number of standard deviations from the middle band for the upper/lower bands."},
    )
    exit_mode: str = field(
        default=MIDDLE_BAND_REVERT,
        metadata={"description": f"Exit rule. Supported: {sorted(SUPPORTED_EXIT_MODES)}."},
    )

    def validate(self) -> None:
        if self.window < 1:
            raise ValueError("window must be positive.")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive.")
        if self.exit_mode not in SUPPORTED_EXIT_MODES:
            raise ValueError(
                f"Unsupported exit_mode: {self.exit_mode!r}. Supported: {sorted(SUPPORTED_EXIT_MODES)}."
            )


def _compute_bands(adj_close: pd.Series, window: int, num_std: float) -> pd.DataFrame:
    """Compute middle/upper/lower Bollinger Bands, parameterized by ``window``/``num_std``.

    Mirrors ``indicators._compute_symbol_indicators``'s Bollinger Band logic
    exactly, generalized the same way ``SmaCrossoverStrategy`` and
    ``RsiMeanReversionStrategy`` generalize their own indicators: computed
    here rather than read from ``indicators_daily``, which only holds one
    fixed (20, 2.0) parameterization and can't serve a parameter sweep over
    arbitrary windows/multipliers (see ``validate_strategy.py``).
    """
    middle = adj_close.rolling(window=window, min_periods=window).mean()
    std = adj_close.rolling(window=window, min_periods=window).std()
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": middle + num_std * std,
            "bb_lower": middle - num_std * std,
        }
    )


@register_strategy("bollinger_breakout")
class BollingerBreakoutStrategy(Strategy):
    """Bollinger Band breakout: buy strength above the upper band, exit on reversion.

    Emits a BUY when price crosses from at/below the upper band to above it
    (a breakout) and a SELL per ``config.exit_mode`` (currently only
    ``'middle_band_revert'``: price crossing back below the middle band,
    i.e. momentum fading back to the mean). Only crossing *events* are
    emitted — no HOLD rows.

    Bands are computed here, directly from ``adj_close`` at whatever
    ``window``/``num_std`` ``config`` specifies — not read from
    ``indicators_daily``, for the same reason the other strategies in this
    package compute their own indicators (see ``_compute_bands``).

    ``exit_mode`` is a dispatch point, not a hardcoded branch buried in the
    signal logic: ``_compute_exit_mask`` is the one place that reads it, so
    adding a second mode (e.g. ``'below_upper_band'``, exiting only when
    price falls back below the upper band rather than the middle) means
    adding one more ``elif`` there and to ``SUPPORTED_EXIT_MODES`` — nothing
    else in this class needs to change.
    """

    base_name = "bollinger_breakout"
    config_cls = BollingerBreakoutConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    config: BollingerBreakoutConfig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect per-symbol upper-band breakout / exit crossings.

        Args:
            df: Merged market data with at least ``symbol``, ``date``,
                ``adj_close`` for one or more symbols. Should include full
                per-symbol history (not just the window of interest) so the
                bands are properly warmed up by the time signals are
                filtered to whatever range the caller actually wants.

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
        """Compute this symbol's bands and detect breakout-entry/exit crossings."""
        data = group.copy()
        bands = _compute_bands(data["adj_close"], self.config.window, self.config.num_std)
        data[["bb_middle", "bb_upper", "bb_lower"]] = bands

        data = data.loc[data["bb_middle"].notna() & data["bb_upper"].notna() & data["bb_lower"].notna()]
        if len(data) < 2:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        prev_close = data["adj_close"].shift(1)
        prev_middle = data["bb_middle"].shift(1)
        prev_upper = data["bb_upper"].shift(1)
        valid_event = prev_close.notna() & prev_middle.notna() & prev_upper.notna()

        buy_mask = valid_event & (prev_close <= prev_upper) & (data["adj_close"] > data["bb_upper"])
        sell_mask = self._compute_exit_mask(data, prev_close, prev_middle, prev_upper, valid_event)

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
                    "reason": "Close broke above upper Bollinger Band (breakout entry)",
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
                    "reason": "Close reverted below middle Bollinger Band (exit)",
                }
            )

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.DataFrame(rows)

    def _compute_exit_mask(
        self,
        data: pd.DataFrame,
        prev_close: pd.Series,
        prev_middle: pd.Series,
        prev_upper: pd.Series,
        valid_event: pd.Series,
    ) -> pd.Series:
        """Compute the SELL/exit crossing mask for ``self.config.exit_mode``.

        The extension point for new exit modes: each branch gets whatever
        prev/current columns it needs (all already computed above) and
        returns a boolean crossing mask aligned with ``data``'s index.
        """
        if self.config.exit_mode == MIDDLE_BAND_REVERT:
            return valid_event & (prev_close >= prev_middle) & (data["adj_close"] < data["bb_middle"])

        # Future mode, e.g.:
        # if self.config.exit_mode == "below_upper_band":
        #     return valid_event & (prev_close >= prev_upper) & (data["adj_close"] < data["bb_upper"])

        raise ValueError(
            f"Unsupported exit_mode: {self.config.exit_mode!r}. "
            f"Supported: {sorted(SUPPORTED_EXIT_MODES)}"
        )

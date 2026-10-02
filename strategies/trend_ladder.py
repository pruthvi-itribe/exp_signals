"""Trend Ladder strategy: buy a fresh 20-EMA reclaim inside a fully stacked uptrend, exit fast below it.

Source: "Trend Ladder Strategy" spec (Chartink scanner + entry/exit/laddering
rules + a 13-year, Nifty 500, published backtest). This module implements the
per-symbol entry/exit *signal* logic exactly as specified. Two parts of the
full written strategy are still deliberately NOT implemented here —
documented below and in ``TrendLadderStrategy``'s docstring, not silently
dropped:

1. **Laddering** (2-3 entries per symbol on successive re-triggers). The
   spec's own published backtest numbers are for "one entry per trigger with
   no laddering" — and ``backtest.py``'s shared simulation engine already
   skips a BUY signal for a symbol with an open position, for every
   strategy. So the *tested* configuration is what this repo already does
   for free; laddering itself would need the engine extended to track
   multiple concurrent lots per symbol.
2. **Risk-based/allocation-capped position sizing** (2-3% risk per trade,
   10-15% cap per stock) and a **separate hard percentage stop** distinct
   from the 20-EMA exit. `backtest.py` only implements
   `position_sizing='equal_weight'` and has no percentage-stop concept at
   all — both are properties of the shared simulation engine, not of any
   one strategy's signal logic.

**The Nifty 50 market-regime filter is now implemented** ("no new entries
while the index is below its own 100 EMA; exit everything on a specific
index-level breakdown pattern") — see ``src.market_regime`` for how the
index's own data reaches `generate_signals` (broadcast onto every symbol's
row by date, via `src.strategy.load_strategy_input` /
`validate_strategy._load_ohlcv_history`) and for the exact, explicitly
flagged interpretation this uses for "breakdown pattern" (the source spec
never pins that down precisely; this repo treats it as the index's close
crossing from at/above its own 100-EMA to below it — mirroring the entry
gate as an exit trigger). The filter is *inert* — behaves exactly as before
this feature existed — whenever the ``nifty_regime_bullish`` /
``nifty_regime_breakdown`` columns aren't present on the input at all (e.g.
existing callers/tests that build a DataFrame directly without going
through either loader above), so this is additive, not a breaking change to
`required_columns`.

None of this changes the core BUY/SELL trigger logic itself, which follows
the spec's 11 scanner conditions, momentum-candle/doji check, and 20-EMA
ladder-rung trigger exactly. The two remaining omissions above mean a
backtest run against this strategy in this repo still won't fully reproduce
the spec's own 23.5% CAGR / 27.6% max drawdown / 1.32 Sharpe numbers — which
the spec itself already flags as an upper bound (survivorship bias from
using today's Nifty 500 list, no costs/taxes deducted, fills at the exact
close) — but should track it much more closely now that the regime filter,
the single biggest driver of the gap, is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy


@dataclass(frozen=True)
class TrendLadderConfig(StrategyConfig):
    """Tunable parameters for :class:`TrendLadderStrategy`.

    Defaults match the spec's Chartink scanner and entry/exit rules exactly
    (EMA 10/20/50/100/200, ADX(14) > 15, volume > 1.2x its 20-day average,
    three-days-running higher closes).
    """

    higher_close_lookback: int = field(
        default=3,
        metadata={"description": "Today's close must exceed the close this many days back, for each day back to 1 (the 'N days running' momentum check)."},
    )
    ema_fast: int = field(
        default=10, metadata={"description": "Fastest EMA in the stack; today's close must be above it."}
    )
    ema_20: int = field(
        default=20,
        metadata={"description": "The ladder-rung/exit EMA -- price must reclaim it from below to enter, and a bearish close below it exits."},
    )
    ema_50: int = field(default=50, metadata={"description": "Third EMA in the required ascending stack (fast > 20 > 50 > 100 > 200)."})
    ema_100: int = field(default=100, metadata={"description": "Fourth EMA in the required ascending stack."})
    ema_200: int = field(default=200, metadata={"description": "Slowest EMA in the required ascending stack."})
    adx_period: int = field(default=14, metadata={"description": "Lookback period for the ADX trend-strength indicator."})
    adx_threshold: float = field(
        default=15.0, metadata={"description": "Minimum ADX to confirm directional strength; rules out sideways chop."}
    )
    volume_window: int = field(default=20, metadata={"description": "Lookback window for the average-volume baseline."})
    volume_multiplier: float = field(
        default=1.2, metadata={"description": "Today's volume must exceed this multiple of its recent average to confirm participation."}
    )
    min_body_ratio: float = field(
        default=0.2,
        metadata={
            "description": (
                "Minimum candle body size, as a fraction of the day's high-low range, to count as a "
                "'solid body, not a doji' bull candle. The spec names this qualitatively without a "
                "number; 0.2 is this implementation's threshold, not part of the original 11 scanner "
                "conditions -- lowered from an initial guess of 0.3 after checking it was the sole "
                "blocking condition on real reclaim-day setups (e.g. GRANULES 2015-09-18, body ratio "
                "0.23) that a looser doji filter would have let through."
            )
        },
    )

    def validate(self) -> None:
        emas = [self.ema_fast, self.ema_20, self.ema_50, self.ema_100, self.ema_200]
        if emas != sorted(emas) or len(set(emas)) != len(emas):
            raise ValueError(f"EMA periods must be strictly increasing from ema_fast to ema_200; got {emas}.")
        if self.higher_close_lookback < 1:
            raise ValueError("higher_close_lookback must be positive.")
        if self.adx_period < 1:
            raise ValueError("adx_period must be positive.")
        if self.adx_threshold < 0:
            raise ValueError("adx_threshold must be non-negative.")
        if self.volume_window < 1:
            raise ValueError("volume_window must be positive.")
        if self.volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive.")
        if not (0.0 <= self.min_body_ratio <= 1.0):
            raise ValueError("min_body_ratio must be between 0 and 1.")


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average Directional Index, Wilder-style smoothing via EMA(alpha=1/period).

    Not read from ``indicators_daily`` -- that table doesn't compute ADX at
    all; this strategy is the first thing in the repo that needs it, so it's
    self-contained here, mirroring how every other strategy in this package
    computes its own indicators rather than depending on a shared table.

    True range and +DM/-DM are smoothed with ``ewm(alpha=1/period,
    adjust=False)`` rather than Wilder's original running-sum recurrence --
    a very close, widely used approximation with the same steady-state
    behavior. +DI/-DI are derived from those smoothed series, and ADX is the
    same smoothing applied to the resulting DX series.
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)
    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]

    smoothing = dict(alpha=1.0 / period, adjust=False, min_periods=period)
    smoothed_tr = true_range.ewm(**smoothing).mean()
    smoothed_plus_dm = plus_dm.ewm(**smoothing).mean()
    smoothed_minus_dm = minus_dm.ewm(**smoothing).mean()

    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

    # float('nan'), not pd.NA: replacing into a float64 Series with pd.NA
    # upcasts it to object dtype the moment any row has zero net directional
    # movement (a real case, e.g. two candles with tied high/low ranges),
    # which then makes the ewm(...).mean() below raise
    # pandas.errors.DataError instead of quietly propagating NaN.
    di_sum = (plus_di + minus_di).replace(0, float("nan"))
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum

    return dx.ewm(**smoothing).mean()


@register_strategy("trend_ladder")
class TrendLadderStrategy(Strategy):
    """Buy a fresh 20-EMA reclaim inside a fully stacked uptrend; exit on a bearish close below it.

    BUY when, on the same day: today's close exceeds each of the last
    ``higher_close_lookback`` days' closes; today is a bull candle (close >
    open) with a solid, non-doji body; price is above the fast EMA and every
    EMA from ``ema_fast`` to ``ema_200`` is stacked in ascending order; ADX
    confirms trend strength; volume confirms participation above its recent
    average; today is specifically the day price reclaims the 20 EMA from
    below (yesterday's close was at/below it) — a fresh ladder rung, not day
    40 of an already-extended move; AND the Nifty 50 market-regime filter is
    bullish (see ``src.market_regime`` — inert if that data isn't attached
    to the input). SELL when a bearish candle's close falls below the 20
    EMA, OR unconditionally for every symbol on the one day the market
    regime filter flags as a breakdown ("Nifty filter exit-all", the same
    reason string the source spec's own tooling uses) — ``backtest.py``'s
    engine already ignores a SELL for a symbol with no open position, so
    this needs no per-symbol "is a position open" tracking here. Only
    crossing/trigger *events* are emitted, matching every other strategy in
    this package.

    See this module's docstring for what's deliberately not implemented
    (laddering, risk-based sizing/hard stops) and why — those are
    simplifications relative to the full written strategy, not bugs.

    EMA/momentum calculations use ``adj_close`` (repo convention, avoids
    phantom jumps at split/bonus dates); candlestick-shape checks (bull
    candle, doji filter) use the raw ``open``/``high``/``low``/``close``,
    since shape-on-the-day is what those describe, not a long-run-adjusted
    proxy. ADX uses ``high``/``low`` scaled by that day's ``adj_close /
    close`` ratio, so it stays consistent with the adjusted price series
    used for the trend/EMA checks rather than showing artificial spikes on
    raw, unadjusted split days.
    """

    base_name = "trend_ladder"
    config_cls = TrendLadderConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "open", "high", "low", "close", "adj_close", "volume")

    config: TrendLadderConfig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect per-symbol Trend Ladder entry/exit events.

        Args:
            df: Merged market data with at least ``required_columns`` for
                one or more symbols. Should include full per-symbol history
                (not just the window of interest) so the EMA(200)/ADX/volume
                indicators are properly warmed up by the time signals are
                filtered to whatever range the caller actually wants.

        Returns:
            Signal rows for trigger days only, matching ``SIGNAL_OUTPUT_COLUMNS``.
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
        """Compute this symbol's indicators and detect entry/exit trigger events."""
        cfg = self.config
        data = group.copy()

        price = data["adj_close"]
        raw_open = data["open"]
        raw_high = data["high"]
        raw_low = data["low"]
        raw_close = data["close"]
        volume = data["volume"]

        # Same-day adj_close/close ratio, applied to raw high/low, so ADX stays
        # consistent with the adjusted close series rather than raw high/low,
        # which can spike artificially on stock-split days.
        adj_ratio = (price / raw_close.replace(0, pd.NA)).fillna(1.0)
        adj_high = raw_high * adj_ratio
        adj_low = raw_low * adj_ratio

        data["ema_fast"] = price.ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
        data["ema_20"] = price.ewm(span=cfg.ema_20, adjust=False, min_periods=cfg.ema_20).mean()
        data["ema_50"] = price.ewm(span=cfg.ema_50, adjust=False, min_periods=cfg.ema_50).mean()
        data["ema_100"] = price.ewm(span=cfg.ema_100, adjust=False, min_periods=cfg.ema_100).mean()
        data["ema_200"] = price.ewm(span=cfg.ema_200, adjust=False, min_periods=cfg.ema_200).mean()
        data["adx"] = _compute_adx(adj_high, adj_low, price, cfg.adx_period)
        data["volume_avg"] = volume.rolling(window=cfg.volume_window, min_periods=cfg.volume_window).mean()

        required_indicator_cols = ["ema_fast", "ema_20", "ema_50", "ema_100", "ema_200", "adx", "volume_avg"]
        valid = data[required_indicator_cols].notna().all(axis=1)
        for k in range(1, cfg.higher_close_lookback + 1):
            valid &= price.shift(k).notna()
        data = data.loc[valid].copy()
        if len(data) < 2:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        # Re-slice the raw/adjusted series to the same filtered rows for the mask logic below.
        price = data["adj_close"]
        raw_open = data["open"]
        raw_high = data["high"]
        raw_low = data["low"]
        raw_close = data["close"]
        volume = data["volume"]

        higher_closes = pd.Series(True, index=data.index)
        for k in range(1, cfg.higher_close_lookback + 1):
            higher_closes &= price > price.shift(k)

        bull_candle = raw_close > raw_open
        above_fast_ema = price > data["ema_fast"]
        ema_stack = (
            (data["ema_fast"] > data["ema_20"])
            & (data["ema_20"] > data["ema_50"])
            & (data["ema_50"] > data["ema_100"])
            & (data["ema_100"] > data["ema_200"])
        )
        strong_trend = data["adx"] > cfg.adx_threshold
        volume_confirmed = volume > (data["volume_avg"] * cfg.volume_multiplier)

        candle_range = (raw_high - raw_low).replace(0, pd.NA)
        body_ratio = ((raw_close - raw_open).abs() / candle_range).fillna(0)
        solid_body = body_ratio >= cfg.min_body_ratio

        prev_price = price.shift(1)
        prev_ema_20 = data["ema_20"].shift(1)
        valid_event = prev_price.notna() & prev_ema_20.notna()
        reclaimed_ema_20 = valid_event & (prev_price <= prev_ema_20) & (price > data["ema_20"])

        # Nifty 50 market-regime filter (see src.market_regime) -- inert
        # (True / False, no effect) when these columns aren't attached to
        # the input at all, so this is additive over existing callers.
        if "nifty_regime_bullish" in data.columns:
            nifty_bullish = data["nifty_regime_bullish"].fillna(True)
        else:
            nifty_bullish = pd.Series(True, index=data.index)
        if "nifty_regime_breakdown" in data.columns:
            nifty_breakdown = data["nifty_regime_breakdown"].fillna(False)
        else:
            nifty_breakdown = pd.Series(False, index=data.index)

        buy_mask = (
            higher_closes
            & bull_candle
            & above_fast_ema
            & ema_stack
            & strong_trend
            & volume_confirmed
            & solid_body
            & reclaimed_ema_20
            & nifty_bullish
        )

        # Crossing event, not a persistent state: yesterday's close was at/above
        # the 20 EMA, today's is a bearish candle closing below it. Without the
        # "yesterday >= EMA20" gate this would re-fire on every bearish day
        # during an extended stretch already below the EMA, which the shared
        # backtest engine happens to no-op (no open position to close) but
        # would still bloat the signals table and violate the "events, not a
        # HOLD/state row for every day" convention every strategy here follows.
        # The Nifty-breakdown exit fires unconditionally (every symbol, not
        # just ones with an EMA-20 cross that day) -- excluded from the
        # "normal" mask so a symbol never gets two SELL rows on the same date.
        bearish_candle = raw_close < raw_open
        normal_sell_mask = (
            valid_event & (prev_price >= prev_ema_20) & bearish_candle & (price < data["ema_20"])
        ) & ~nifty_breakdown
        sell_mask = normal_sell_mask | nifty_breakdown

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
                    "reason": "Reclaimed 20 EMA within full EMA stack, ADX and volume confirmed (Trend Ladder entry)",
                }
            )

        for idx in data.index[sell_mask]:
            row = data.loc[idx]
            reason = (
                "Nifty filter exit-all"
                if nifty_breakdown.loc[idx]
                else "Bearish candle closed below 20 EMA (Trend Ladder exit)"
            )
            rows.append(
                {
                    "symbol": row["symbol"],
                    "date": row["date"],
                    "strategy": self.name,
                    "signal_type": "SELL",
                    "price": float(row["adj_close"]),
                    "reason": reason,
                }
            )

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.DataFrame(rows)

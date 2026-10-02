"""Precision Pullback strategy: buy the confirmed resumption of an established uptrend
after it dips to the 50 EMA and bounces; exit when the 50 EMA gives way.

Source: "Precision Pullback Strategy" spec (a colour-coded 50-EMA "band" +
a six-step entry sequence + a published 13-year Nifty 500 backtest). This
module implements the per-symbol entry/exit *signal* logic as specified.
One part of the full written strategy is still deliberately NOT implemented
here — documented below and in ``PrecisionPullbackStrategy``'s docstring,
not silently dropped (see ``strategies/README.md`` for the consolidated
list across every strategy in this package):

1. **The one-re-entry-per-cycle rule** ("within 20 calendar days of a stop,
   if the band is blue again and a bull candle closes above the *previous
   trade's high*, re-enter once"). This needs per-trade state — specifically
   the highest price reached *while a specific trade was open* — that
   `generate_signals` has no way to know: signals are generated once, up
   front, for a symbol's whole history, independent of and before the
   backtest simulation that later decides which signals actually open or
   close a position. There is no clean way to ask "what was the high of the
   trade this SELL signal is closing?" from inside signal generation alone.
   This is part of the spec's own *tested* configuration (unlike Trend
   Ladder's omitted laddering, which was NOT part of its tested config), so
   omitting it is expected to cause more divergence from the published
   backtest than Trend Ladder's omissions did — flagged here for exactly
   that reason.

**The Nifty 50 market-regime filter is now implemented** — identical
mechanism and interpretation to Trend Ladder's (see
`strategies/trend_ladder.py` and `src.market_regime` for the full
explanation, including the explicitly-flagged interpretation of "breakdown
pattern"). Same inert-if-absent contract: the ``nifty_regime_bullish`` /
``nifty_regime_breakdown`` columns are optional, not part of
`required_columns`, so this is additive over existing callers/tests.

**Risk-based/allocation-capped position sizing** and a separate hard
percentage stop are also not implemented, again matching Trend Ladder:
`backtest.py` only implements `position_sizing='equal_weight'` and has no
percentage-stop concept, both properties of the shared simulation engine
rather than of any one strategy.

None of this changes the core BUY/SELL trigger logic, which follows the
spec's six-step qualify -> pullback -> recovery -> mark -> continuation
sequence and the 50-EMA exit exactly. The remaining omission above means a
backtest run against this strategy in this repo still won't fully reproduce
the spec's own 19.4% CAGR / 13.7% max drawdown / 1.59 Sharpe numbers, which
the spec itself already flags as an upper bound (survivorship bias from
using today's Nifty 500 list, no costs/taxes deducted, fills at the exact
close) — but should track it more closely now that the regime filter is
active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig, first_exit_after_each_buy
from strategies.registry import register_strategy


@dataclass(frozen=True)
class PrecisionPullbackConfig(StrategyConfig):
    """Tunable parameters for :class:`PrecisionPullbackStrategy`.

    Defaults match the spec exactly: a 50-day EMA, and a 90-trading-day
    unbroken "blue" streak required before the strategy starts watching for
    a pullback at all.
    """

    ema_period: int = field(default=50, metadata={"description": "The single moving average the whole strategy runs on."})
    blue_days_required: int = field(
        default=90,
        metadata={"description": "Consecutive trading days closing at/above the EMA required to qualify a stock's uptrend before a pullback setup is even watched for."},
    )

    def validate(self) -> None:
        if self.ema_period < 1:
            raise ValueError("ema_period must be positive.")
        if self.blue_days_required < 1:
            raise ValueError("blue_days_required must be positive.")


class _State(Enum):
    """Per-symbol state machine stages, in the order the spec describes them."""

    COUNTING_BLUE = auto()          # step 1: accumulating an unbroken blue streak toward blue_days_required
    WAITING_FOR_RED = auto()        # qualified; waiting for the first red close (the pullback signal)
    WAITING_FOR_RECOVERY = auto()   # red seen; waiting for a bull candle whose full body is back above the band
    WAITING_FOR_PULLBACK_MARK = auto()  # recovery candle seen; tracking the highest high until a down-day freezes the mark
    WAITING_FOR_CONTINUATION = auto()   # mark frozen; waiting for a bull candle to close above it (the entry)


@register_strategy("precision_pullback")
class PrecisionPullbackStrategy(Strategy):
    """Buy the confirmed resumption of an established uptrend at the 50 EMA; exit when it gives way.

    Unlike every other strategy in this package, this one is a genuine
    multi-day state machine, not an independent per-row crossing condition
    — whether today is a valid entry depends on a specific sequence having
    already played out over the preceding weeks or months, not just on
    today's own indicator values. So instead of vectorized boolean masks,
    signal generation here is a single sequential scan per symbol
    (`_scan_symbol`), advancing one of five states each day:

      1. ``COUNTING_BLUE`` — count an unbroken run of "blue" days (close at
         or above the 50 EMA; a close exactly on the line, "gray" in the
         spec's own colour language, counts as blue too, never as a reset).
         Any red close (close below the EMA) resets the count to zero. Once
         the count reaches ``blue_days_required``, the uptrend is
         considered established.
      2. ``WAITING_FOR_RED`` — idle until the first red close: the pullback.
      3. ``WAITING_FOR_RECOVERY`` — waiting for a bull candle (close > open)
         whose *entire body* — open and close both — sits above the EMA.
         Red days here are explicitly tolerated per the spec ("a pullback
         can take several sessions to finish") and do not reset anything.
      4. ``WAITING_FOR_PULLBACK_MARK`` — from the recovery candle onward,
         track the running highest high. The first day whose close is below
         the *previous* day's close marks the start of a small pullback;
         the running highest high as of the day *before* that (not
         including it) is frozen as "the mark." Any red close during this
         stage resets all the way back to state 1 — a stricter reset than
         state 3's, per the spec.
      5. ``WAITING_FOR_CONTINUATION`` — the mark is fixed; wait for a bull
         candle whose close exceeds it. That close is the BUY. A red close
         here also resets to state 1.

    After a BUY, the state resets to ``COUNTING_BLUE`` — the next setup for
    that symbol needs a fresh qualifying streak, matching the spec's own
    "wait for a fresh 90-day blue cycle" language for what normally happens
    after a trade (the one exception, "one re-entry per cycle," is a
    documented omission — see this module's docstring).

    SELL fires on the first bearish candle (close < open) closing below the
    EMA after each BUY -- not only on the day price crossed the EMA, so a
    gap below on a bull candle followed by a bearish close still exits --
    OR unconditionally for every symbol on the one day the Nifty 50
    market-regime filter flags as a breakdown ("Nifty filter exit-all"; see
    `src.market_regime`, inert if that data isn't attached to the input).
    `backtest.py`'s engine ignores a SELL for a symbol with no open
    position. The entry-side
    continuation trigger (state 5, below) additionally requires the regime
    filter to be bullish that day.

    Matching Trend Ladder's precedent: EMA/trend-level comparisons
    (blue/red classification, the running highest-high/mark tracking) use
    ``adj_close`` and a same-day-ratio-adjusted high, so a 12-year count
    isn't corrupted by a single artificial split-day price jump; pure
    candlestick-shape checks (bull/bear direction, "full body above the
    band") use the raw ``open``/``close``, since shape-on-the-day is what
    those describe.
    """

    base_name = "precision_pullback"
    config_cls = PrecisionPullbackConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "open", "high", "close", "adj_close")

    config: PrecisionPullbackConfig

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect per-symbol Precision Pullback entry/exit events.

        Args:
            df: Merged market data with at least ``required_columns`` for
                one or more symbols. Should include full per-symbol history
                (not just the window of interest) so the EMA and the blue-day
                count are properly warmed up, and so a qualifying streak
                that began before the window of interest is still honored.

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
            symbol_signals = self._scan_symbol(group.reset_index(drop=True))
            if not symbol_signals.empty:
                signal_frames.append(symbol_signals)

        if not signal_frames:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.concat(signal_frames, ignore_index=True).loc[:, SIGNAL_OUTPUT_COLUMNS]

    def _scan_symbol(self, group: pd.DataFrame) -> pd.DataFrame:
        """Sequential per-day state-machine scan for one symbol. See class docstring."""
        cfg = self.config

        adj_close = group["adj_close"]
        raw_open = group["open"].to_numpy()
        raw_close = group["close"].to_numpy()
        raw_high = group["high"].to_numpy()

        ema = adj_close.ewm(span=cfg.ema_period, adjust=False, min_periods=cfg.ema_period).mean().to_numpy()
        price = adj_close.to_numpy()

        # Same-day adj_close/close ratio applied to raw high, so the highest-high/mark
        # tracking stays on the same adjusted basis as the EMA line and blue/red
        # classification, rather than raw high, which can spike on split days.
        adj_ratio = (adj_close / group["close"].replace(0, pd.NA)).fillna(1.0).to_numpy()
        adj_high = raw_high * adj_ratio

        # Nifty 50 market-regime filter (see src.market_regime) -- inert
        # (True / False, no effect) when these columns aren't attached to
        # the input at all, so this is additive over existing callers.
        if "nifty_regime_bullish" in group.columns:
            nifty_bullish = group["nifty_regime_bullish"].fillna(True).to_numpy()
        else:
            nifty_bullish = [True] * len(group)
        if "nifty_regime_breakdown" in group.columns:
            nifty_breakdown = group["nifty_regime_breakdown"].fillna(False).to_numpy()
        else:
            nifty_breakdown = [False] * len(group)

        symbol = group["symbol"].iloc[0]
        dates = group["date"].to_numpy()
        n = len(group)

        rows: list[dict[str, object]] = []
        buy_days = [False] * n

        state = _State.COUNTING_BLUE
        blue_streak = 0
        running_max_high: float | None = None
        mark: float | None = None

        for i in range(n):
            if pd.isna(ema[i]):
                continue  # EMA warm-up: no state progression possible yet

            is_blue = price[i] >= ema[i]          # gray (touching) counts as blue, never resets
            is_red = price[i] < ema[i]
            is_bull = raw_close[i] > raw_open[i]
            is_bear = raw_close[i] < raw_open[i]

            if state is _State.COUNTING_BLUE:
                if is_red:
                    blue_streak = 0
                else:
                    blue_streak += 1
                    if blue_streak >= cfg.blue_days_required:
                        state = _State.WAITING_FOR_RED
                continue

            if state is _State.WAITING_FOR_RED:
                if is_red:
                    state = _State.WAITING_FOR_RECOVERY
                continue

            if state is _State.WAITING_FOR_RECOVERY:
                # Full body above the band: both open and close above the EMA line
                # (strictly above -- "gray"/touching doesn't count as a recovery).
                if is_bull and raw_open[i] > ema[i] and price[i] > ema[i]:
                    state = _State.WAITING_FOR_PULLBACK_MARK
                    running_max_high = adj_high[i]
                # Red days (and any other non-qualifying day) are tolerated -- keep waiting.
                continue

            if state is _State.WAITING_FOR_PULLBACK_MARK:
                if is_red:
                    state = _State.COUNTING_BLUE
                    blue_streak = 0
                    running_max_high = None
                    continue
                if price[i] < price[i - 1]:
                    # Down day: freeze the mark using the running max as of *before* today.
                    mark = running_max_high
                    state = _State.WAITING_FOR_CONTINUATION
                else:
                    running_max_high = max(running_max_high, adj_high[i])
                continue

            if state is _State.WAITING_FOR_CONTINUATION:
                if is_red:
                    state = _State.COUNTING_BLUE
                    blue_streak = 0
                    mark = None
                    continue
                if is_bull and mark is not None and price[i] > mark and nifty_bullish[i]:
                    buy_days[i] = True
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": dates[i],
                            "strategy": self.name,
                            "signal_type": "BUY",
                            "price": float(price[i]),
                            "reason": "Confirmed continuation above marked pullback high after 50 EMA bounce (Precision Pullback entry)",
                        }
                    )
                    state = _State.COUNTING_BLUE
                    blue_streak = 0
                    mark = None
                continue

        # SELL: a bearish candle closing below the EMA -- the first such day after
        # each BUY, not only the day price crossed the EMA (a gap below on a bull
        # candle followed by a bearish close is still an exit). One SELL per
        # entry keeps signals as events rather than a row per bearish day. The
        # Nifty breakdown exit fires unconditionally for every symbol on that one
        # day (and never doubles up with a normal SELL on the same date).
        exit_condition = [
            not pd.isna(ema[i]) and raw_close[i] < raw_open[i] and price[i] < ema[i] for i in range(n)
        ]
        normal_exits = first_exit_after_each_buy(buy_days, exit_condition, nifty_breakdown)
        for i in range(n):
            if nifty_breakdown[i]:
                rows.append(
                    {
                        "symbol": symbol,
                        "date": dates[i],
                        "strategy": self.name,
                        "signal_type": "SELL",
                        "price": float(price[i]),
                        "reason": "Nifty filter exit-all",
                    }
                )
            elif normal_exits[i]:
                rows.append(
                    {
                        "symbol": symbol,
                        "date": dates[i],
                        "strategy": self.name,
                        "signal_type": "SELL",
                        "price": float(price[i]),
                        "reason": "Bearish candle closed below 50 EMA (Precision Pullback exit)",
                    }
                )

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        result = pd.DataFrame(rows)
        return result.sort_values("date").reset_index(drop=True)

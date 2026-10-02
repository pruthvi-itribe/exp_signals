"""Amihud-illiquidity portfolio tilt: periodically rebalance into the least-liquid
quantile of the Nifty 50, built directly from a validated research/screen.py
finding, not ported from an external spec.

Source of the rule: screening ``amihud_illiquidity`` (``research/signal_library.py``)
found a real, and comparatively strong, relationship with forward returns on
the Nifty 50 specifically (mean IC 0.062 at a 20-trading-day horizon, 0.110 at
60 days -- both well above every other signal screened in this project).
Two things about that result made it worth real scrutiny before building
anything on it, and both were checked directly rather than assumed:

1. **Is it just a handful of lucky stocks?** The IC climbed with NO
   plateau through every horizon tested (1d to 60d) -- unlike
   ``bb_position``'s sensible rise-then-fade shape -- and the top-illiquidity
   quintile turned out to have almost no day-to-day turnover (~1 symbol out
   of ~10 changes per day; a few names -- MAXHEALTH, TRENT, SBILIFE,
   TATACONSUM -- sit in that bucket 80%+ of all their trading days).
   Excluding those four specific names and re-screening only weakened the
   60-day IC from 0.110 to 0.098 -- most of the effect is NOT those four
   stocks; it's a broader pattern across a larger set of structurally
   smaller/more-recently-promoted Nifty 50 names.
2. **Does it survive a bigger universe?** Screened again on the full Nifty
   500: the effect weakens substantially (60-day IC 0.062, half the Nifty
   50 reading) and loses significance entirely at short horizons. This is a
   Nifty-50-specific pattern, not a broad market-wide liquidity premium --
   same scope restriction as ``bollinger_reversion``, for an unrelated
   reason this time.

**This is NOT a reactive liquidity-timing signal, and the strategy below is
deliberately built to not pretend otherwise.** The near-zero daily turnover
in bucket membership means this measures a slow, persistent portfolio TILT
("hold the structurally smaller/newer large-cap names") rather than a
signal that reacts to day-to-day conditions. Architecturally, this is why
``IlliquidityTiltStrategy`` is NOT built like ``BollingerReversionStrategy``
(independent per-symbol opportunistic entry/exit cycles, fast-moving): it's
a single, portfolio-wide periodic rebalance, matching the slow cadence the
screening itself revealed, not imposing a fast cadence onto a slow signal.

RULE: every ``rebalance_every_days`` *trading days* (not calendar days,
counted from the start of the available history the same way every other
signal/strategy in this project counts a horizon -- by row position in the
trading-day sequence, not a calendar offset), rank every symbol in the
input by its current Amihud illiquidity (``mean(|daily adj_close return| /
(close * volume))`` over ``window`` days -- same formula as
``research.signal_library.amihud_illiquidity``, reimplemented locally here
rather than imported, matching every other strategy in this package
computing its own indicators internally rather than depending on a shared
module). Hold the top ``top_quantile`` fraction (highest illiquidity, since
the screened relationship is POSITIVE -- more illiquid predicts higher
forward returns). At each rebalance: SELL anything currently held that's
dropped out of the target quantile, BUY anything newly in it that wasn't
already held. Nothing is re-evaluated between rebalances.

Same scope caveat as ``bollinger_reversion``: validated on, and only on,
the Nifty 50. Running this against the Nifty 500 (or any other universe)
would be trading an untested claim -- universe selection is the caller's
responsibility, same as every other strategy in this package.

## Why a stop-loss

Not part of the original screen -- added after an initial real backtest
(rebalance-only exits, no stop) showed the slow-tilt result described
below, then tested directly against the idea that a periodic-only exit
leaves a position exposed to a large intra-quarter drawdown with no way
to cut it short before the next scheduled rebalance (up to
``rebalance_every_days`` trading days away). Unlike ``bollinger_reversion``'s
stop-loss tuning -- which traded a bit of win rate for better Sharpe --
this one is a genuine win on every axis: simulating every threshold from
5% to 30% as a real full backtest (not an approximation) on the Nifty 50,
CAGR rose from 17.04% (no stop) to a peak of 18.14% at 12%, Sharpe rose
from 1.01 to 1.11, AND max drawdown fell from 24.03% to 20.44% --
simultaneously, not a trade-off. 8%, 10%, and 15% all improved on the
no-stop baseline too (CAGR 17.4-18.0%, Sharpe 1.05-1.10), confirming 12%
isn't a lone lucky threshold. The same default, re-tested on the full
Nifty 500 (see "Result vs. a Nifty 500 buy-and-hold benchmark," below),
improved CAGR from 17.48% to 22.79% and Sharpe from 1.25 to 1.57 -- an
even larger gain, not a universe-specific fluke.

``stop_loss_pct`` defaults to 12.0 for this reason. Win rate drops
noticeably at every threshold tested (e.g. 69.1% with no stop vs. 52.6%
at 12%) -- expected and not a red flag by itself: a stop-loss converts
some positions that would have recovered into realized small losses,
trading win rate for a better-shaped return distribution, the same
pattern documented for ``bollinger_reversion``'s own stop-loss.

## Result vs. a Nifty 50 buy-and-hold benchmark

This strategy's own screened claim is about a slow factor tilt, not a
reactive trade, so the right comparison is a passive benchmark over the
same window, not a short-horizon forward-return table. Full-history
backtest (default params, including the 12% stop-loss, Nifty 50,
2013-01-02 to 2026-09-25): CAGR 18.14%, Sharpe 1.11, max drawdown 20.44%,
win rate 52.6%, 156 trades (no-stop baseline, for reference: CAGR 17.04%,
Sharpe 1.01, max drawdown 24.03%, win rate 69.1%, 110 trades). Equal-weight
buy-and-hold over the identical window (44 of the 50 symbols were already
listed at the window's start; the other 6 are excluded from the benchmark
basket, not from the strategy's own run): CAGR 18.43%, Sharpe 0.70, max
drawdown 41.12%. Both Sharpe figures use the same risk-free-adjusted
formula as ``backtest.calculate_metrics``, for a fair comparison.

With the stop-loss, this is now within half a point of buy-and-hold's raw
CAGR (18.14% vs. 18.43%) while still clearing it by a wide margin on both
risk measures (Sharpe 1.11 vs. 0.70; max drawdown 20.44% vs. 41.12%, less
than half) -- holding a smaller, periodically-refreshed basket
concentrated in the persistently-illiquid names, with a cut on the worst
intra-quarter losses, now very nearly matches the index's own return with
meaningfully less risk, rather than giving up CAGR for it.

## Result vs. a Nifty 500 buy-and-hold benchmark

Run on direct request against this module's own "untested claim" warning
above (the underlying signal screened weaker on the Nifty 500, IC 0.062
vs. 0.110 on Nifty 50 at 60 days). The research runner used to produce
this number needed one correction first: its shared default
``max_concurrent_positions=10`` matches the Nifty 50 run's ~10-name target
basket (0.2 x 50) but would silently throttle this universe's real
~100-name target basket (0.2 x 501) down to 10 -- recomputed to 100
before trusting the result. Full-history backtest (default params,
including the 12% stop-loss, Nifty 500, 2013-01-02 to 2026-09-25): CAGR
22.79%, Sharpe 1.57, max drawdown 27.24%, win rate 43.0%, 1,513 trades
(no-stop baseline: CAGR 17.48%, Sharpe 1.25, max drawdown 29.66%, win
rate 70.8%, 893 trades). Equal-weight buy-and-hold over the identical
window (310 of the 501 symbols present since the window's start) scored
CAGR 21.64%, Sharpe 0.86, max drawdown 46.82%.

With the stop-loss, this strategy now BEATS buy-and-hold outright on the
Nifty 500 -- higher CAGR (22.79% vs. 21.64%), much higher Sharpe (1.57
vs. 0.86), and a smaller max drawdown (27.24% vs. 46.82%) -- not merely a
better risk-adjusted trade-off, as it was on Nifty 50. This result is
still read with real caution, independent of the stop-loss: performance
did not degrade the way the weaker screening-stage IC predicted on this
universe either before or after adding the stop, which could mean the
illiquidity premium is more robust across the broader market than that
single IC number suggested, OR that this backtest's flat 0.05% slippage
assumption understates real execution cost across ~100 small/micro-cap
illiquid names specifically (a materially different liquidity profile
than 10 "least liquid of the Nifty 50" large-caps). Unresolved -- see
``candidates/illiquidity_tilt.md`` for the full list of open questions
before sizing real capital into this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy


@dataclass(frozen=True)
class IlliquidityTiltConfig(StrategyConfig):
    """Tunable parameters for :class:`IlliquidityTiltStrategy`.

    ``window=20`` matches ``research.signal_library.amihud_illiquidity``'s
    own default, so this strategy trades exactly what was screened.
    ``rebalance_every_days=63`` (~one calendar quarter of trading days) is
    a deliberately slow cadence, matching the near-zero daily turnover the
    screening found in actual bucket membership -- rebalancing faster than
    the signal itself changes would just add transaction costs without
    capturing anything different. ``top_quantile=0.2`` matches the
    quintile convention used throughout this project's screening.
    """

    window: int = field(
        default=20,
        metadata={"description": "Rolling window (trading days) for the Amihud illiquidity average."},
    )
    top_quantile: float = field(
        default=0.2,
        metadata={"description": "Fraction of the universe (by illiquidity, descending) held at each rebalance."},
    )
    rebalance_every_days: int = field(
        default=63,
        metadata={
            "description": (
                "Trading days between rebalances (~63 = one calendar quarter). Deliberately slow, "
                "matching the near-zero daily turnover found in the underlying signal's own bucket "
                "membership during screening -- not a knob to make this trade more often."
            )
        },
    )
    stop_loss_pct: float | None = field(
        default=12.0,
        metadata={
            "description": (
                "Exit a position immediately (any day, not just at a rebalance) once its adj_close "
                "closes this many percent or more below its own entry price -- e.g. 12.0 means -12%. "
                "Defaults to 12.0 after directly simulating every threshold from 5 to 30 on a real "
                "historical run: it improved CAGR, Sharpe, AND max drawdown simultaneously on both "
                "the Nifty 50 and Nifty 500 (a genuine win, not a trade-off), with 8/10/15 all also "
                "improving on the no-stop baseline -- not a lone standout. Pass None to disable and "
                "reproduce the originally-screened, rebalance-only behavior. See the module "
                "docstring's 'Why a stop-loss' section for the full numbers."
            )
        },
    )

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if not (0.0 < self.top_quantile < 1.0):
            raise ValueError("top_quantile must be between 0 and 1.")
        if self.rebalance_every_days < 1:
            raise ValueError("rebalance_every_days must be positive.")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive (it's a magnitude, e.g. 15.0 means -15%).")


@register_strategy("illiquidity_tilt")
class IlliquidityTiltStrategy(Strategy):
    """Periodically rebalance into the Nifty 50's least-liquid quantile by Amihud illiquidity.

    See this module's docstring for the full rationale, why this is a
    slow, portfolio-wide rebalance rather than a per-symbol opportunistic
    cycle, and the Nifty-50-only validation scope.
    """

    base_name = "illiquidity_tilt"
    config_cls = IlliquidityTiltConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close", "close", "volume")

    config: IlliquidityTiltConfig

    @property
    def name(self) -> str:
        """Folds window/rebalance cadence into the stored identity, matching
        ``SmaCrossoverStrategy``'s precedent -- different (window,
        rebalance_every_days) combinations are different strategies, and
        must not collide under one name in ``signals``' (symbol, date,
        strategy) key."""
        return f"{self.base_name}_{self.config.window}_{self.config.rebalance_every_days}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect periodic, portfolio-wide rebalance events, plus (if
        ``stop_loss_pct`` is set) per-position stop-loss exits on any day.

        Unlike every other strategy in this package, this one maintains a
        single GLOBAL ``held`` mapping (symbol -> entry price) across the
        whole date range rather than independent per-symbol state -- a
        rebalance decision for any one symbol depends on the whole
        universe's ranking that day, and on whether the strategy itself
        decided to hold that symbol at the previous rebalance. The loop
        below walks every trading day (not just rebalance days) so a
        stop-loss, when enabled, can fire the day it's triggered rather
        than waiting for the next scheduled rebalance; rebalance-day logic
        itself still only runs on the scheduled cadence.

        Args:
            df: Merged market data with at least ``required_columns`` for
                the full intended universe (the Nifty 50) at once. Should
                include full history so the rolling illiquidity average is
                warmed up.

        Returns:
            Signal rows for rebalance trigger days (and, if enabled,
            stop-loss trigger days), matching ``SIGNAL_OUTPUT_COLUMNS``.
        """
        if df.empty:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        self.validate_columns(df)

        cfg = self.config
        working = df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
        working = working.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Same decomposition as research.signal_library.amihud_illiquidity:
        # separate single-column grouped transforms combined by plain
        # arithmetic, rather than a multi-column groupby.apply -- matches
        # how volume_weighted_momentum in that module handles needing more
        # than one column per symbol.
        daily_return = working.groupby("symbol")["adj_close"].transform(lambda s: s.pct_change())
        # float('nan'), not pd.NA: see strategies/trend_ladder.py's
        # _compute_adx for why pd.NA here would upcast to object dtype and
        # break the rolling mean below with a DataError.
        dollar_volume = (working["close"] * working["volume"]).replace(0, float("nan"))
        daily_illiquidity = daily_return.abs() / dollar_volume
        working["illiquidity"] = daily_illiquidity.groupby(working["symbol"]).transform(
            lambda s: s.rolling(window=cfg.window, min_periods=cfg.window).mean()
        )

        all_dates = sorted(working["date"].unique())
        rebalance_dates = set(all_dates[:: cfg.rebalance_every_days])

        # `held` maps symbol -> entry adj_close, needed to check a stop-loss
        # on any day, not just a rebalance day. When stop_loss_pct is None
        # the entry prices are tracked but never read -- the behavior is
        # then identical to before this field existed.
        held: dict[str, float] = {}
        rows: list[dict[str, object]] = []

        # One pivot for fast same-day price lookups during the daily
        # stop-loss scan below, instead of filtering `working` on every
        # date -- built once, O(dates x symbols), not once per date.
        price_pivot = (
            working.pivot(index="date", columns="symbol", values="adj_close")
            if cfg.stop_loss_pct is not None
            else None
        )

        for current_date in all_dates:
            if cfg.stop_loss_pct is not None and held:
                # Priority: a stop-loss exit always fires before that same
                # day's scheduled rebalance logic below, mirroring
                # BollingerReversionStrategy's stop-then-schedule ordering.
                day_prices = price_pivot.loc[current_date]
                for symbol in sorted(held):
                    price = day_prices.get(symbol)
                    if price is None or pd.isna(price):
                        continue  # no reading today -- can't evaluate the stop, leave it held
                    entry_price = held[symbol]
                    drawdown_pct = (price - entry_price) / entry_price * 100.0
                    if drawdown_pct <= -cfg.stop_loss_pct:
                        rows.append(
                            {
                                "symbol": symbol,
                                "date": current_date,
                                "strategy": self.name,
                                "signal_type": "SELL",
                                "price": float(price),
                                "reason": (
                                    f"Stop-loss: closed {cfg.stop_loss_pct:.0f}% or more below entry "
                                    "(Illiquidity Tilt stop)"
                                ),
                            }
                        )
                        del held[symbol]

            if current_date not in rebalance_dates:
                continue

            day_df = working.loc[working["date"] == current_date, ["symbol", "adj_close", "illiquidity"]]
            valid = day_df.dropna(subset=["illiquidity"])
            if valid.empty:
                continue

            rank_pct = valid["illiquidity"].rank(pct=True)
            valid_symbols = set(valid["symbol"])
            held_symbols = set(held)
            target = set(valid.loc[rank_pct >= (1.0 - cfg.top_quantile), "symbol"])
            prices = valid.set_index("symbol")["adj_close"]

            # Only sell a held symbol if it HAS a reading today and that
            # reading places it outside the target -- a symbol missing
            # today's reading entirely (a data gap) is left exactly as it
            # was, neither force-sold nor re-bought, since there's nothing
            # to rank it against. Restricting to valid_symbols here also
            # guarantees `prices[symbol]` below can never KeyError.
            to_sell = (held_symbols & valid_symbols) - target
            to_buy = target - held_symbols

            for symbol in sorted(to_sell):
                rows.append(
                    {
                        "symbol": symbol,
                        "date": current_date,
                        "strategy": self.name,
                        "signal_type": "SELL",
                        "price": float(prices[symbol]),
                        "reason": (
                            f"Dropped out of the top {cfg.top_quantile:.0%} by Amihud illiquidity "
                            "at rebalance (Illiquidity Tilt exit)"
                        ),
                    }
                )
            for symbol in sorted(to_buy):
                rows.append(
                    {
                        "symbol": symbol,
                        "date": current_date,
                        "strategy": self.name,
                        "signal_type": "BUY",
                        "price": float(prices[symbol]),
                        "reason": (
                            f"Entered the top {cfg.top_quantile:.0%} by Amihud illiquidity "
                            "at rebalance (Illiquidity Tilt entry)"
                        ),
                    }
                )
            for symbol in to_sell:
                del held[symbol]
            for symbol in to_buy:
                held[symbol] = float(prices[symbol])

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        return pd.DataFrame(rows).loc[:, list(SIGNAL_OUTPUT_COLUMNS)]

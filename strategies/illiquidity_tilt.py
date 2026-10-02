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

## Result vs. a Nifty 50 buy-and-hold benchmark

This strategy's own screened claim is about a slow factor tilt, not a
reactive trade, so the right comparison is a passive benchmark over the
same window, not a short-horizon forward-return table. Full-history
backtest (default params, Nifty 50, 2013-01-02 to 2026-09-25): CAGR
17.04%, Sharpe 1.01, max drawdown 24.03%, win rate 69.1%, 110 trades.
Equal-weight buy-and-hold over the identical window (44 of the 50 symbols
were already listed at the window's start; the other 6 are excluded from
the benchmark basket, not from the strategy's own run): CAGR 18.43%,
Sharpe 0.70, max drawdown 41.12%. Both Sharpe figures use the same
risk-free-adjusted formula as ``backtest.calculate_metrics``, for a fair
comparison.

Reads exactly as the slow-tilt hypothesis predicted: this does NOT beat
buy-and-hold on raw CAGR (17.04% vs. 18.43%, a real but small gap), but it
clears it by a wide margin on both risk measures (Sharpe 1.01 vs. 0.70;
max drawdown 24.03% vs. 41.12%, nearly half) -- holding a smaller,
periodically-refreshed basket concentrated in the persistently-illiquid
names gave up a little upside for a meaningfully smoother ride, rather
than producing an outright CAGR edge over the index.
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

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if not (0.0 < self.top_quantile < 1.0):
            raise ValueError("top_quantile must be between 0 and 1.")
        if self.rebalance_every_days < 1:
            raise ValueError("rebalance_every_days must be positive.")


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
        """Detect periodic, portfolio-wide rebalance entry/exit events.

        Unlike every other strategy in this package, this one maintains a
        single GLOBAL ``held`` set across the whole date range rather than
        independent per-symbol state -- a rebalance decision for any one
        symbol depends on the whole universe's ranking that day, and on
        whether the strategy itself decided to hold that symbol at the
        previous rebalance.

        Args:
            df: Merged market data with at least ``required_columns`` for
                the full intended universe (the Nifty 50) at once. Should
                include full history so the rolling illiquidity average is
                warmed up.

        Returns:
            Signal rows for rebalance trigger days only, matching
            ``SIGNAL_OUTPUT_COLUMNS``.
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
        rebalance_dates = all_dates[:: cfg.rebalance_every_days]

        held: set[str] = set()
        rows: list[dict[str, object]] = []

        for rebal_date in rebalance_dates:
            day_df = working.loc[working["date"] == rebal_date, ["symbol", "adj_close", "illiquidity"]]
            valid = day_df.dropna(subset=["illiquidity"])
            if valid.empty:
                continue

            rank_pct = valid["illiquidity"].rank(pct=True)
            valid_symbols = set(valid["symbol"])
            target = set(valid.loc[rank_pct >= (1.0 - cfg.top_quantile), "symbol"])
            prices = valid.set_index("symbol")["adj_close"]

            # Only sell a held symbol if it HAS a reading today and that
            # reading places it outside the target -- a symbol missing
            # today's reading entirely (a data gap) is left exactly as it
            # was, neither force-sold nor re-bought, since there's nothing
            # to rank it against. Restricting to valid_symbols here also
            # guarantees `prices[symbol]` below can never KeyError.
            to_sell = (held & valid_symbols) - target
            to_buy = target - held

            for symbol in sorted(to_sell):
                rows.append(
                    {
                        "symbol": symbol,
                        "date": rebal_date,
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
                        "date": rebal_date,
                        "strategy": self.name,
                        "signal_type": "BUY",
                        "price": float(prices[symbol]),
                        "reason": (
                            f"Entered the top {cfg.top_quantile:.0%} by Amihud illiquidity "
                            "at rebalance (Illiquidity Tilt entry)"
                        ),
                    }
                )
            held = (held - to_sell) | to_buy

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        return pd.DataFrame(rows).loc[:, list(SIGNAL_OUTPUT_COLUMNS)]

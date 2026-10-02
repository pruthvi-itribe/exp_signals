"""Bollinger-band cross-sectional mean-reversion strategy, built directly from
a validated research/screen.py finding, not ported from an external spec.

Source of the entry rule: screening ``bb_position`` (``research/signal_library.py``)
with ``research/screen.py`` showed a real, if weak, mean-reversion relationship
that is **specific to large-cap stocks** and **does not survive being applied
to the full Nifty 500** -- the mid/small-cap half of the market shows the
opposite pattern (a fading, sign-flipping relationship), and blending the two
in one screen washes the effect out to statistical noise. See the large-cap
vs. rest comparison and the window/horizon sweeps that motivated this
strategy's exact parameters (``window=30``, ``holding_period_days=30``) --
those were the empirically strongest, most stable combination found, not
arbitrary defaults.

THIS IS A DIFFERENT KIND OF STRATEGY than every other one in this package:
every other ``Strategy`` subclass processes each symbol's own price history
independently and reacts to a single-symbol price-pattern crossing event.
This one is a genuine **cross-sectional, relative-value** rule -- "buy
whichever stocks are currently cheapest *relative to the rest of the
universe passed in*," which only means anything when the full intended
universe is scored together in one ``generate_signals`` call. Concretely:

- Running this against a single symbol (e.g. ``backtest_cli.py run --symbol X``)
  will never produce a BUY at all: ranking one stock against itself always
  gives it the 100th percentile, which can never fall inside
  ``bottom_quantile``. This is expected, not a bug.
- The validated universe is the Nifty 50 (large-cap) specifically. Running
  this against the Nifty 500 (or any other universe) re-introduces exactly
  the cancellation effect the screening found -- nothing here enforces that
  restriction; it is the caller's responsibility (``--universe nifty50`` /
  an explicit large-cap symbol list), the same way every other strategy in
  this package leaves universe selection to the caller.

ENTRY: on each date, rank every symbol in the input by its current
``bb_position`` (``(adj_close - lower_band) / (upper_band - lower_band)``,
bands from a simple ``window``-day rolling mean/std -- the conventional
Bollinger Band definition, matching ``research/signal_library.py`` exactly
so this strategy trades precisely what was screened, not an approximation
of it). A symbol not already in an active holding cycle that lands in the
bottom ``bottom_quantile`` fraction of that day's cross-section gets a BUY.

EXIT: whichever comes first of two conditions. (1) A fixed
``holding_period_days`` *trading days* after entry (not calendar days --
counted the same way ``research/forward_returns.py`` counts a
forward-return horizon, by row position in that symbol's own trading-day
sequence), regardless of what its rank has done in the meantime -- this
half is a deliberate, direct translation of what the screen actually
measured ("if I bought on day T and held for N days, what would I have
made"), NOT "hold until it reverts to the middle band," which would be a
different, untested claim. (2) A per-position stop-loss: if price closes
``stop_loss_pct`` or more below the entry price at ANY point during the
hold, exit immediately at that close -- see ``StopLossConfig``'s field and
"Why a stop-loss," below, for where this number came from and why it's a
deviation from the literal screened rule, same as the entry rule's "while
in a cycle, don't re-evaluate" clause below.

While a symbol is in its holding cycle, it is not re-evaluated for a fresh
entry even if it stays in the bottom quantile the whole time -- this avoids
firing a new BUY signal every single day a cheap stock remains cheap, which
`backtest.py`'s one-position-per-symbol rule would mostly no-op anyway, but
would still bloat the ``signals`` table against the "events, not a state
row for every day" convention every strategy in this package follows.

## Why a stop-loss

(The numbers below were measured with backtest engine v1, which sized each
entry at ``cash / N``; the 15% stop has not been re-tuned under v2. See
PERFORMANCE.md for v2 results.)

Not part of the original screen -- added after an initial real backtest
(fixed-horizon exit only) showed a real per-trade edge (53.5% win rate,
average win exceeding average loss) undermined by a weak Sharpe ratio
(0.06) and a 22.3% max drawdown, traced to a handful of large, clustered
losses landing almost exactly on known market-stress windows (the COVID
crash, the Jan 2023 Adani/Hindenburg selloff) -- the fixed 30-day exit gave
a falling-knife trade no way to cut a loss short before its scheduled day.
A Nifty-index-level regime filter was tried FIRST for this exact problem
and reverted after measuring it live -- see "Nifty 50 market-regime
filter," below, for why a *per-position, price-based* cut was tried next
instead of a market-wide one.

``stop_loss_pct`` defaults to 15 (meaning: exit if price falls 15% or more
below entry), chosen by directly simulating every historical trade from
that same backtest run: for each trade, exit at the first day its price
closed below various threshold levels instead of waiting for the scheduled
exit, and compare the resulting total P&L against what actually happened.
-5% made things worse (total P&L dropped by ~Rs 180,700 -- too tight, it
cuts off trades that would have recovered to a profit). -8% through -20%
all improved on the no-stop baseline; -15% was the best of those tested
(+Rs 66,500 versus the actual no-stop-loss result). This simulation ignores
one real effect in either direction: it doesn't account for capital freed
up by an earlier exit being redeployed into a fresh trade sooner, which the
actual backtest run below will capture and this approximation couldn't.

## Nifty 50 market-regime filter

No Nifty 50 market-regime filter (``src.market_regime``) is wired in here.
It WAS tried -- entry gated on the index's own 100-EMA, plus an exit-all on
a breakdown day -- after an initial real backtest run (no filter) showed a
real per-trade edge (53.5% win rate, average win exceeding average loss)
undermined by a weak Sharpe ratio and a 22%+ drawdown, driven by a handful
of large, clustered losses landing almost exactly on known market-stress
windows (the COVID crash, the Jan 2023 Adani/Hindenburg selloff). It was
reverted after measuring it, not on theory: it made things worse, not
better (932 trades/53.5% win/0.06 Sharpe/22.3% drawdown became 1,275
trades/44.0% win/-0.04 Sharpe/17.0% drawdown -- drawdown genuinely improved,
but win rate collapsed and Sharpe went negative). The root cause: the
index's own 100-EMA crosses ~89 times across this backtest's ~12 years --
roughly once every 53 trading days, close enough to this strategy's own
30-day holding period that nearly every position gets intercepted by a
breakdown before reaching its originally-screened, validated exit, making
the filter the *dominant* exit mechanism (1,415 exit-all closes vs. 1,167
normal ones in that run) rather than a rare protective override. This is
also the third strategy in this package where adding this exact filter
measurably hurt risk-adjusted returns (`trend_ladder`: roughly flat CAGR,
worse drawdown; `precision_pullback`: worse CAGR and Sharpe) -- the
consistent culprit looks like the breakdown definition itself being too
frequent/noisy to function as a genuine regime signal, not something
specific to any one strategy's mechanics. Worth revisiting with a slower
breakdown definition (e.g. requiring it to persist several days, or a
longer EMA) before trying this again, rather than assuming the same
single-day 100-EMA cross will behave differently here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy, StrategyConfig
from strategies.registry import register_strategy


@dataclass(frozen=True)
class BollingerReversionConfig(StrategyConfig):
    """Tunable parameters for :class:`BollingerReversionStrategy`.

    Defaults (``window=30``, ``holding_period_days=30``) are the empirically
    strongest, most stable combination found during screening -- see this
    module's docstring. ``bottom_quantile=0.2`` matches the quintile
    (``n_buckets=5``) convention ``research.decile_analysis`` used by
    default, so "bottom quantile" here means exactly the same bucket the
    screen's decile-spread numbers describe.
    """

    window: int = field(
        default=30,
        metadata={"description": "Rolling window (trading days) for the Bollinger middle band and std dev."},
    )
    num_std: float = field(
        default=2.0,
        metadata={
            "description": (
                "Band width in standard deviations. Note: under this strategy's bottom_quantile "
                "ranking (a rank-based, not absolute-threshold, entry rule), num_std does not change "
                "which symbols qualify -- rescaling the band width is a positive linear transform of "
                "bb_position for a fixed symbol/date, which never changes cross-sectional rank order. "
                "It's kept as a real parameter (rather than removed) because a future absolute-threshold "
                "variant, or switching to Pearson-based screening, would make it matter again."
            )
        },
    )
    bottom_quantile: float = field(
        default=0.2,
        metadata={"description": "Fraction of the day's cross-section (by bb_position, ascending) eligible for entry."},
    )
    holding_period_days: int = field(
        default=30,
        metadata={"description": "Fixed holding period in trading days (not calendar days) before the exit fires."},
    )
    stop_loss_pct: float = field(
        default=15.0,
        metadata={
            "description": (
                "Exit immediately if price closes this many percent or more below entry, before the "
                "fixed holding period would otherwise end. 15.0 was the best-performing level found by "
                "directly simulating this exact threshold's effect on every trade from a real historical "
                "run -- see 'Why a stop-loss' in this module's docstring for the full calibration."
            )
        },
    )

    def validate(self) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2.")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive.")
        if not (0.0 < self.bottom_quantile < 1.0):
            raise ValueError("bottom_quantile must be between 0 and 1.")
        if self.holding_period_days < 1:
            raise ValueError("holding_period_days must be positive.")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive (it's a magnitude, e.g. 15.0 means -15%).")


@register_strategy("bollinger_reversion")
class BollingerReversionStrategy(Strategy):
    """Buy the cheapest cross-sectional quantile by Bollinger-band position; hold a fixed period.

    See this module's docstring for the full rationale, the cross-sectional
    (not per-symbol-independent) nature of the entry rule, the large-cap-only
    validation scope, and why a Nifty 50 market-regime filter was tried and
    then reverted after measuring it live.
    """

    base_name = "bollinger_reversion"
    config_cls = BollingerReversionConfig
    required_columns: tuple[str, ...] = ("symbol", "date", "adj_close")

    config: BollingerReversionConfig

    @property
    def name(self) -> str:
        """Folds window/holding-period into the stored identity, matching
        ``SmaCrossoverStrategy``'s precedent -- different (window,
        holding_period) combinations are different strategies, and must not
        collide under one name in ``signals``' (symbol, date, strategy) key."""
        return f"{self.base_name}_{self.config.window}_{self.config.holding_period_days}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect cross-sectional entry events and their fixed-horizon exits.

        Args:
            df: Merged market data with at least ``required_columns`` for
                the FULL intended universe at once (see this module's
                docstring on why a single-symbol call never buys anything).
                Should include full history so the rolling band is warmed up.

        Returns:
            Signal rows for trigger days only, matching ``SIGNAL_OUTPUT_COLUMNS``.
        """
        if df.empty:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        self.validate_columns(df)

        cfg = self.config
        working = df.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
        working = working.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Cross-sectional step, done once over the whole panel: each symbol's
        # own bb_position (its own rolling band, computed independently),
        # then ranked against every OTHER symbol on the same date. This
        # mirrors research.signal_library.bb_position and
        # research.ic_analysis's cross-sectional-by-date setup exactly.
        def _bb_position(price: pd.Series) -> pd.Series:
            middle = price.rolling(window=cfg.window, min_periods=cfg.window).mean()
            std = price.rolling(window=cfg.window, min_periods=cfg.window).std()
            upper = middle + cfg.num_std * std
            lower = middle - cfg.num_std * std
            return (price - lower) / (upper - lower)

        working["bb_position"] = working.groupby("symbol")["adj_close"].transform(_bb_position)
        rank_pct = working.groupby("date")["bb_position"].rank(pct=True)
        working["in_bottom_quantile"] = working["bb_position"].notna() & (rank_pct <= cfg.bottom_quantile)

        signal_frames: list[pd.DataFrame] = []
        for _, group in working.groupby("symbol", sort=True):
            symbol_signals = self._generate_symbol_signals(group.reset_index(drop=True))
            if not symbol_signals.empty:
                signal_frames.append(symbol_signals)

        if not signal_frames:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))

        return pd.concat(signal_frames, ignore_index=True).loc[:, SIGNAL_OUTPUT_COLUMNS]

    def _generate_symbol_signals(self, group: pd.DataFrame) -> pd.DataFrame:
        """Per-symbol fixed-holding-period cycle (with an early stop-loss
        exit), driven by the pre-computed cross-sectional
        ``in_bottom_quantile`` column. See class/module docstrings for why
        this isn't a simple vectorized mask: a symbol already in an active
        cycle must not re-trigger a fresh entry, and the stop-loss must be
        checked against THIS cycle's own entry price every day it's open."""
        cfg = self.config
        symbol = group["symbol"].iloc[0]
        dates = group["date"].to_numpy()
        price = group["adj_close"].to_numpy()
        in_bottom = group["in_bottom_quantile"].to_numpy()
        n = len(group)

        rows: list[dict[str, object]] = []
        in_cycle = False
        entry_idx: int | None = None
        entry_price: float | None = None

        for i in range(n):
            if in_cycle:
                # Stop-loss takes priority over the scheduled exit -- if
                # both would trigger on the same day, only one SELL is emitted.
                drawdown_pct = (price[i] - entry_price) / entry_price * 100.0
                if drawdown_pct <= -cfg.stop_loss_pct:
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": dates[i],
                            "strategy": self.name,
                            "signal_type": "SELL",
                            "price": float(price[i]),
                            "reason": (
                                f"Stop-loss: closed {cfg.stop_loss_pct:.0f}% or more below entry "
                                "(Bollinger Reversion exit)"
                            ),
                        }
                    )
                    in_cycle = False
                    entry_idx = None
                    entry_price = None
                    continue
                if i == entry_idx + cfg.holding_period_days:
                    rows.append(
                        {
                            "symbol": symbol,
                            "date": dates[i],
                            "strategy": self.name,
                            "signal_type": "SELL",
                            "price": float(price[i]),
                            "reason": (
                                f"Fixed {cfg.holding_period_days}-trading-day holding period elapsed "
                                "(Bollinger Reversion exit)"
                            ),
                        }
                    )
                    in_cycle = False
                    entry_idx = None
                    entry_price = None
                continue

            if in_bottom[i]:
                rows.append(
                    {
                        "symbol": symbol,
                        "date": dates[i],
                        "strategy": self.name,
                        "signal_type": "BUY",
                        "price": float(price[i]),
                        "reason": (
                            f"Entered bottom {cfg.bottom_quantile:.0%} of the universe's Bollinger-band "
                            "position (Bollinger Reversion entry)"
                        ),
                    }
                )
                in_cycle = True
                entry_idx = i
                entry_price = float(price[i])
            # Not in a cycle and not in the bottom quantile today: no row --
            # an unresolved cycle at the end of history is left for
            # backtest.py's own END_OF_BACKTEST force-close to handle, same
            # as every other strategy in this package relies on.

        if not rows:
            return pd.DataFrame(columns=list(SIGNAL_OUTPUT_COLUMNS))
        return pd.DataFrame(rows)

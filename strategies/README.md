# `strategies/` — Registered Trading Strategies

Each concrete strategy is a `Strategy` subclass (`strategies/base.py`) registered
under a string name (`strategies/registry.py`). See `ARCHITECTURE.md` for the
full module-by-module reference and dependency map; this file is specifically
about **fidelity to source specs** — two of the seven registered strategies
(`trend_ladder`, `precision_pullback`) were ported from external PDF write-ups
that include their own published backtest, and each has parts of the written
strategy that are *deliberately not implemented*. This is the single place
that lists every one of those omissions in one table, so they're never
silently assumed away when reading a backtest result off this repo's data —
each is also documented in its strategy's own module docstring, but this file
is the one to check first if you're comparing a run here against a source
spec's published numbers.

## `bollinger_reversion` and `illiquidity_tilt` — not ported from a source spec, so not in the tables below

Unlike the other five, these two weren't ported from an external write-up
— both were built directly from `research/screen.py` findings (see each
one's own module docstring for its full validation trail: `bollinger_reversion`'s
horizon/window sweeps and large-cap-vs-rest split; `illiquidity_tilt`'s
exclude-the-winners and Nifty-500-dilution checks on `amihud_illiquidity`).
There's no external "published" number to compare against or omit from, so
neither has anything to add to the fidelity-tracking tables below.

Both are also architecturally different from the five ported/hand-written
strategies — cross-sectional, relative-value rules rather than a per-symbol
price-pattern trigger — which each one's own docstring covers in detail,
including why running either against a single symbol, or against a
universe wider than the one it was validated on (Nifty 50 for both), either
silently produces a meaningless result (`bollinger_reversion`, a single
symbol never buys) or the opposite (`illiquidity_tilt`, a single symbol
*always* buys) rather than an error either way.

`illiquidity_tilt` is also architecturally different from `bollinger_reversion`
specifically: it rebalances the whole portfolio together on a fixed
calendar-like schedule (driven by one global `held` set) rather than
running independent per-symbol opportunistic entry/exit cycles — a
deliberate match to what screening found (near-zero day-to-day turnover in
the underlying signal's own ranking), not an arbitrary design choice.

## Why omissions happen at all

Every omission below falls into one of two buckets:

1. **The shared backtest engine doesn't support it.** `backtest.py` implements
   one position-sizing scheme (`equal_weight`) and no percentage-based stop
   distinct from a strategy's own exit signal. A source spec's risk-based
   sizing (e.g. "2-3% risk per trade, capped at 10-15% of capital") or a
   separate hard stop are properties of the *simulation engine*, shared by
   every strategy — implementing them properly means extending `backtest.py`
   itself, not adding a flag to one `Strategy` subclass.
2. **The current `Strategy` architecture doesn't support it.** Some rules
   need data `generate_signals(df)` doesn't have. The Nifty 50 market-regime
   filter used to be in this bucket (see "Now implemented," below, for how it
   was bridged without changing `Strategy`'s interface); `precision_pullback`'s
   re-entry rule still is — see its own table entry.

Neither bucket covers bugs in the ported strategies' own trigger logic. One
such bug has since been fixed: both strategies' SELL only fired on the day
price crossed the EMA, so a bearish close below it a day or more later did
not exit until the next qualifying cross or a Nifty exit-all.
`precision_pullback`'s multi-day state machine is unit-tested against
hand-traced synthetic scenarios (see its module for the reasoning). Both
buckets mean a backtest run here will not fully reproduce a source spec's
own published numbers, and shouldn't be read as attempting to.

## Now implemented: the Nifty 50 market-regime filter

Originally omitted here for the same reason as everything else in bucket 2
above: `generate_signals(df)` never received the index's own price history,
so no strategy could see market-wide regime state. The fix (`src/market_regime.py`)
didn't need a `Strategy` interface change or an engine change: the Nifty 50
index's own OHLCV is fetched and stored in `ohlcv_data` like any other symbol
(under `^NSEI`), and its derived regime columns
(`nifty_regime_bullish`, `nifty_regime_breakdown`) are broadcast onto every
*other* symbol's row **by date** (not by symbol) before a strategy ever sees
the data — both `src.strategy.load_strategy_input` and
`validate_strategy._load_ohlcv_history` attach them. Each strategy just reads
two more ordinary input columns.

Two things worth knowing if you're comparing a run against these numbers:

- **The exact "breakdown pattern" is an interpretation, not a verified
  transcription.** Neither this repo's docs nor the source-spec PDFs (not
  present in this repo) pin down that rule's precise technical definition —
  only that entries are gated by "index below its own 100 EMA." This repo
  defines "breakdown" as the day the index's close crosses from at/above its
  100-EMA to below it, mirroring the entry gate as a one-time exit trigger.
  Revisit this if the exact source rule ever surfaces.
- **The filter is inert (no effect) unless attached.** `nifty_regime_bullish`/
  `nifty_regime_breakdown` are optional inputs, not part of either strategy's
  `required_columns` — a caller that builds a DataFrame directly (as every
  existing unit test does) gets the pre-filter behavior unchanged.
- **CAGR/Sharpe/drawdown did not uniformly converge toward the published
  numbers just from this one fix** — see each strategy's updated "Result vs.
  published" line below. Trade count and win rate moved closer (fewer,
  higher-quality entries, as intended); the index's own 100-EMA crosses 89
  times in 13 years (roughly every 2 months), which is frequent enough to
  cause real whipsaw — forcing an exit-all near a temporary trough and
  re-entering only once the regime turns bullish again can realize a loss a
  hold would have recovered from, which is a plausible reason drawdown *rose*
  for both strategies rather than fell. The remaining gap is spread across
  the strategies' other still-omitted pieces (laddering, re-entry,
  risk-based sizing, hard stops) and the survivorship-bias/no-costs
  differences documented below, not concentrated in one remaining cause.

## Omissions by strategy

### `trend_ladder` (source: "Trend Ladder Strategy" PDF)

| Omitted | Why | Bucket |
|---|---|---|
| Laddering (2-3 entries per symbol on successive re-triggers) | The source spec's own *published* backtest is explicitly "one entry per trigger, no laddering" — `backtest.py` already enforces exactly that (skips a BUY for a symbol with an open position) for every strategy, so the *tested* configuration is what this repo does for free. Not a gap versus the numbers actually being compared against. | — (matches tested config) |
| ~~Nifty 50 market-regime filter~~ | **Implemented** — see "Now implemented," above. | — |
| Risk-based/allocation-capped position sizing (2-3% risk/trade, 10-15% cap) | `backtest.py` only implements `equal_weight`. | Engine |
| Separate hard percentage stop distinct from the 20-EMA exit | No percentage-stop concept in the engine at all. | Engine |

**Result vs. published**: this repo, *before* the regime filter — CAGR 9.82%, max drawdown 12.94%, Sharpe 0.43, 1,488 trades, win rate 34.2%. This repo, *after* — CAGR 9.90%, max drawdown 16.68%, Sharpe 0.49, 1,210 trades, win rate 35.2% (full Nifty 500, 2014-09-24 to 2026-09-18). Published — CAGR 23.5%, max drawdown 27.63%, Sharpe 1.32, 994 trades, win rate 42.05%. After the exit, sizing, ordering and cost fixes (re-downloaded data, 2014-10-07 to 2026-09-18) — CAGR 14.94%, max drawdown 28.20%, Sharpe 0.68, 1,228 trades, win rate 34.9%. That figure uses alphabetical order to pick among same-day entries when slots are short, which turns out to be a lucky draw: over 100 random orders the median is 12.1% (90% of runs between 10.5% and 13.7%, none above 14.94%).

### `precision_pullback` (source: "Precision Pullback Strategy" PDF)

| Omitted | Why | Bucket |
|---|---|---|
| One re-entry per cycle after a stop (within 20 calendar days, band blue again, a bull candle closes above the *previous trade's high*) | Needs per-trade state — the highest price reached *while a specific trade was open* — which `generate_signals` cannot know: signals are generated once, up front, for a symbol's whole history, independent of and before the backtest simulation that later decides which signals actually open/close a position. There's no way to ask "what was this SELL's trade's high?" from inside signal generation alone. **This one IS part of the source spec's tested config** (unlike Trend Ladder's laddering), so this omission is expected to cause more divergence from the published backtest than Trend Ladder's omissions did. | Architecture (deeper: needs engine + signal-generation integration, not just one or the other) |
| ~~Nifty 50 market-regime filter~~ | **Implemented** — identical mechanism to `trend_ladder`'s; see "Now implemented," above. | — |
| Risk-based/allocation-capped position sizing | Same as `trend_ladder`. | Engine |
| Separate hard percentage stop | Same as `trend_ladder`. | Engine |

**Result vs. published**: this repo, *before* the regime filter — CAGR 11.98%, max drawdown 11.45%, Sharpe 0.70, 296 trades, win rate 39.2%. This repo, *after* — CAGR 8.64%, max drawdown 12.71%, Sharpe 0.39, 289 trades, win rate 42.2% (full Nifty 500, 2014-09-24 to 2026-09-18) — win rate and trade count moved toward published, but CAGR/Sharpe moved *away*; see "Now implemented," above, for why (whipsaw from re-entering after a forced exit-all is the leading suspect, on top of the still-omitted re-entry rule above). Published — CAGR 19.4%, max drawdown 13.71%, Sharpe 1.59, 258 trades, win rate 44.96%. After the same fixes (2014-10-07 to 2026-09-18) — CAGR 11.51%, max drawdown 19.80%, Sharpe 0.54, 290 trades, win rate 41.7%; almost all of that change comes from the engine fixes (11.44% without the exit fix).

## What both source specs' own published numbers already admit, independent of this repo

Both PDFs disclose the same three caveats about their *own* backtests, which
apply equally to any run of these strategies against this repo's data:

- **Survivorship bias**: both use *today's* Nifty 500 list for the entire
  backtest period, so companies that were removed/delisted/renamed out of the
  index at some point are missing. Both PDFs estimate this overstates CAGR by
  2-5 points. This repo's own Nifty 500 universe has the identical bias — see
  `src/universe.fetch_nifty500_constituents`'s docstring and
  `SURVIVORSHIP_BIAS_WARNING`.
- **No costs/taxes deducted** in the source backtests. This repo's
  `backtest.py` *does* model Indian equity delivery transaction costs
  (STT, exchange charges, stamp duty, a per-sell DP charge, GST) and slippage — so on this specific
  point, this repo's numbers are more conservative than the source PDFs',
  not less.
- **Fills at the exact daily close** in the source backtests. This repo's
  engine fills at the next day's open with slippage instead, which is
  achievable for a scanner run after the close.

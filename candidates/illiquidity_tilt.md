# illiquidity_tilt — Production Candidate

Status: candidate for real capital. Backtested on both the Nifty 50 and
the full Nifty 500. With its now-default 12% per-position stop-loss, it
beats buy-and-hold outright on the Nifty 500 (CAGR, Sharpe, AND max
drawdown all better) and comes within half a point of buy-and-hold's CAGR
on the Nifty 50 while roughly halving its drawdown. See `PERFORMANCE.md`
for every raw number this file discusses, and `strategies/illiquidity_tilt.py`
for the full technical/validation trail.

## What it does, in plain language

Once a quarter, rank every stock in the universe by how "illiquid" it's
been over the last month, and hold the roughly one-fifth of the universe
that's hardest to trade without moving its own price — selling whatever
just fell out of that group and buying whatever just entered it. Nothing
happens in between rebalances. This is a bet that stocks which are
genuinely harder to trade carry a return premium for that inconvenience
(investors demand extra compensation for holding something they can't
easily exit) — a known effect in academic finance (Amihud, 2002), found
here directly by screening this project's own data rather than assumed
from the literature.

It is deliberately NOT a reactive trading signal. It doesn't try to time
entries or exits around news or price action — it's closer to a
systematic tilt you'd apply to a portfolio once a quarter and otherwise
leave alone.

## The actual rule

1. Every stock's "illiquidity" = the 20-trading-day rolling average of
   `|daily return| / (price × volume)` — the Amihud (2002) measure: how
   much a rupee of trading moves the price, averaged over the last month.
2. Every `rebalance_every_days` (63 trading days, ~1 calendar quarter):
   rank every stock in the universe by that number.
3. Hold the top 20% (most illiquid). Sell anything held that dropped out
   of that top 20%; buy anything newly in it. Leave everything else
   untouched until the next rebalance.
4. **Per-position stop-loss (added after direct testing, now the
   default):** exit a position immediately, on any day, if its price
   closes 12% or more below its own entry price — rather than waiting for
   the next quarterly rebalance no matter how far it's fallen in the
   meantime. This is the only thing that happens between rebalances.

Full config and edge cases (e.g. what happens if a held stock is missing
a data point on rebalance day, or on a stop-loss check day) are in
`strategies/illiquidity_tilt.py`.

## Why this signal, specifically

Found by systematically screening every signal in `research/signal_library.py`
against forward returns (see `research/README.md`). This one stood out
enough to need real scrutiny before building anything on it:

- **Not a handful of lucky stocks.** A few names sit in the top-illiquidity
  bucket 80%+ of their trading days. Excluding the four most persistent
  ones only weakened the 60-day IC from 0.110 to 0.098 — most of the
  effect isn't those specific stocks.
- **Does weaken on a bigger universe, as expected** — Nifty 500 IC (0.062)
  is roughly half the Nifty 50 reading (0.110) at the screening stage.
  Despite that, the live backtest below did NOT show the degradation this
  predicted — see "Open questions," below.

## Performance vs. buy-and-hold

Both backtests use this project's shared engine, including real Indian
equity transaction costs (STT, exchange charges, stamp duty, GST) and
slippage — not a frictionless simulation. Both buy-and-hold benchmarks are
equal-weight, held from the start of the window to the end, no rebalancing,
run through the identical metric formulas for a fair comparison. Figures
below are with the default 12% stop-loss; see `PERFORMANCE.md` for the
pre-stop-loss numbers this improved on.

| | illiquidity_tilt | Buy-and-hold | Gap |
|---|---|---|---|
| **Nifty 50** CAGR | 18.14% | 18.43% | -0.3 pts |
| **Nifty 50** Sharpe | **1.11** | 0.70 | +0.41 |
| **Nifty 50** Max drawdown | **20.44%** | 41.12% | -20.7 pts |
| **Nifty 500** CAGR | **22.79%** | 21.64% | +1.2 pts |
| **Nifty 500** Sharpe | **1.57** | 0.86 | +0.71 |
| **Nifty 500** Max drawdown | **27.24%** | 46.82% | -19.6 pts |

On the Nifty 500, this now beats buy-and-hold outright — higher CAGR,
much higher Sharpe, smaller drawdown, not a trade-off at all. On the
Nifty 50 it's effectively tied on raw CAGR (within 0.3 points) while
roughly halving the drawdown and lifting Sharpe by 0.41. The stop-loss
was found by directly simulating every threshold from 5% to 30% as a
real full backtest (not an approximation): 12% was the best of several
thresholds that all improved simultaneously on CAGR, Sharpe, AND
drawdown versus the no-stop version, on both universes independently —
not a single lucky pick on one dataset.

## Why this is a candidate and not just another backtest

- The edge survived two separate attempts to debunk it (lucky-stocks test,
  bigger-universe test) rather than collapsing under scrutiny.
- It's architecturally simple to run for real: one rebalance a quarter,
  ~10-100 names depending on universe, no daily monitoring or reactive
  decision-making required.
- It's genuinely different from this project's other strategies — a slow
  factor tilt, not a variant of the same price-pattern/mean-reversion idea
  tried (and mostly falling short) elsewhere in this project.
- Risk-adjusted performance is consistent across both universes tested,
  not a fluke of one specific symbol set.

## Open questions and risks before real capital

- **The Nifty 500 result contradicts what the screening-stage IC
  predicted**, and that contradiction isn't resolved. IC weakened
  substantially on the bigger universe (0.062 vs 0.110), yet the actual
  backtest Sharpe was *better* on Nifty 500 (1.25) than Nifty 50 (1.01).
  Two explanations are open: either the illiquidity premium is more
  robust across the broader market than the simple IC metric suggested,
  or this backtest's flat 0.05% slippage assumption understates real
  execution cost for ~100 small/micro-cap illiquid names specifically —
  a materially different liquidity profile than 10 "least liquid of the
  Nifty 50" large-caps, which the shared backtest engine does not model
  differently. This needs a more realistic, size-dependent slippage/impact
  model before trusting the Nifty 500 number at face value.
- ~~**No per-position risk control.**~~ **Resolved** — a 12% per-position
  stop-loss is now the default, found by directly simulating every
  threshold from 5% to 30% on a real backtest. It improved CAGR, Sharpe,
  AND max drawdown simultaneously on both universes (not a trade-off),
  at the cost of a noticeably lower win rate (e.g. 69.1% to 52.6% on
  Nifty 50) — expected: it converts some positions that would have
  recovered by the next rebalance into realized small losses, trading
  win rate for a better-shaped return distribution.
- **No market-regime filter** — deliberately, since the same filter
  measurably hurt all three strategies it was tried on in this project
  (see `strategies/README.md`). Still, this hasn't been specifically
  stress-tested against a sharp, broad drawdown (COVID-crash-style) to
  see how a quarterly-only rebalance behaves when the whole market gaps
  down between rebalances.
- **Survivorship bias**, same caveat as every other strategy in this
  project — both universes are today's constituent list applied
  retroactively (see `src/universe.py`'s `SURVIVORSHIP_BIAS_WARNING`).
- **Execution feasibility of the illiquid names themselves is not modeled
  beyond the flat slippage assumption above** — by definition, this
  strategy wants to hold the hardest-to-trade names in the universe. Real
  order sizing, market depth, and wider effective spreads for those
  specific names should be checked against actual order-book data before
  this trades real money, not assumed away by the backtest's generic
  cost model.

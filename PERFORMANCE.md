# Strategy Performance Tracker

Running record of each registered strategy's backtested performance. Updated
as new backtests are run — this is a log, not a one-time snapshot, so keep
adding rows rather than only editing in place, except to correct a row that
turns out to have been measuring the wrong thing (e.g. the wrong universe).

**Engine versions.** Rows tagged **v2** were run with `backtest.ENGINE_VERSION = 2` (equity-based sizing, sells before buys, broker-checked costs; stored in `backtest_runs.engine_version`). Every untagged row was run with the earlier engine (**v1**: each entry sized at `cash / N`, so later entries shrank and much of the capital sat idle). v1 and v2 numbers are not comparable: the same strategy under v2 typically shows a higher CAGR *and* a much deeper drawdown, because more capital is at work. Each v2 row's notes give the v1 result on the same data.

Numbers below were pulled from the persisted `backtest_runs` /
`backtest_results` / `backtest_trades` tables (or, where noted, a research
script's output) as of 2026-10-03 — see each strategy's own module docstring
and `strategies/README.md` for the full methodology/caveats behind a number
before trusting it in isolation.

## Nifty 500 (full universe)

Confirmed via distinct symbols actually traded in the stored run, not just
the label — only these two strategies have a genuine full-Nifty-500 run on
record; see "No Nifty 500 run on record" below for the rest.

| Strategy | CAGR | Sharpe | Win rate | Max drawdown | Trades | Window | Notes |
|---|---|---|---|---|---|---|---|
| `trend_ladder` | 9.90% | 0.49 | 35.2% | 16.68% | 1,210 | 2014-09-24 → 2026-09-18 | — |
| `precision_pullback` | 8.64% | 0.39 | 42.2% | 12.71% | 289 | 2014-09-24 → 2026-09-18 | — |
| `trend_ladder` **v2** (with the exit fix) | 14.94% | 0.68 | 34.9% | 28.20% | 1,228 | 2014-10-07 → 2026-09-18 | Re-downloaded data; this repo's original code scores 10.12% on it. Same-day entries are picked alphabetically when slots are short: over 100 random orders the median is 12.1% (90% between 10.5% and 13.7%). |
| `precision_pullback` **v2** (with the exit fix) | 11.51% | 0.54 | 41.7% | 19.80% | 290 | 2014-10-07 → 2026-09-18 | v1 on the same data: 8.40%, max drawdown 13.75%. |
| `illiquidity_tilt` **v2** (12% stop-loss, 100 slots) | 32.01% | 1.58 | 40.0% | 44.24% | 1,429 | 2014-10-07 → 2026-09-25 | Re-downloaded data; v1 on the same data: 22.39%, max drawdown 30.18%. The 12% stop was tuned under v1 and has not been re-checked under v2. Today's Nifty 500 list for the whole window, so survivorship flatters illiquid small caps most. |
| `illiquidity_tilt` (12% stop-loss, **current default**) | 22.79% | 1.57 | 43.0% | 27.24% | 1,513 | 2013-01-02 → 2026-09-25 | Now BEATS buy-and-hold outright here (CAGR, Sharpe, and drawdown all better) — see the no-stop row below for the pre-stop-loss number this improved on, and "untested claim" caveat that still applies independent of the stop-loss. Benchmark: equal-weight Nifty 500 buy-and-hold (310 symbols present since 2014-09-24) scored CAGR 21.64%, Sharpe 0.86, max drawdown 46.82%. |
| `illiquidity_tilt` (no stop-loss, superseded) | 17.48% | 1.25 | 70.8% | 29.66% | 893 | 2013-01-02 → 2026-09-25 | Run on direct request, against the module docstring's own "untested claim" warning (the underlying `amihud_illiquidity` signal screened weaker here, IC 0.062 vs. 0.110 on Nifty 50 at 60d). **Position cap had to be corrected**: the shared research runner's default `max_concurrent_positions=10` matches the Nifty 50 run's ~10-name target basket (0.2 × 50) but would have throttled this universe's real ~100-name target basket (0.2 × 501) down to 10 — recomputed to 100 before trusting this number. Result was a genuine surprise even before the stop-loss: performance did NOT degrade the way the weaker screening IC predicted (Sharpe 1.25 here vs. 1.01 on Nifty 50) — open question whether that's a real broader effect or an artifact of this backtest's flat 0.05% slippage assumption understating real execution cost across ~100 small/micro-cap illiquid names. |

## Nifty 50

| Strategy | CAGR | Sharpe | Win rate | Max drawdown | Trades | Window | Notes |
|---|---|---|---|---|---|---|---|
| `illiquidity_tilt` **v2** (12% stop-loss, 10 slots) | 30.05% | 1.29 | 50.3% | 36.05% | 157 | 2014-10-07 → 2026-09-25 | Re-downloaded data; v1 on the same data: 21.99%, max drawdown 22.09%. Stop tuned under v1. |
| `bollinger_reversion` **v2** (10 slots) | 9.91% | 0.30 | 51.9% | 41.90% | 983 | 2014-10-07 → 2026-09-25 | Re-downloaded data; v1 on the same data: 6.21%, max drawdown 27.75%. |
| `bollinger_reversion` | 7.45% | 0.20 | 53.6% | 26.04% | 968 | 2014-09-24 → 2026-09-18 | Validated on Nifty 50 only — `bb_position`'s edge reverses sign on the Nifty 500's mid/small-cap half. |
| `illiquidity_tilt` (12% stop-loss, **current default**) | 18.14% | 1.11 | 52.6% | 20.44% | 156 | 2013-01-02 → 2026-09-25 | Chosen by directly simulating every stop threshold from 5% to 30% as a real full backtest: 12% was the best of several thresholds (8/10/15 all also improved on the no-stop baseline, so not a lone lucky pick), improving CAGR, Sharpe, AND max drawdown simultaneously — a genuine win, not a trade-off. Now within half a point of buy-and-hold's own CAGR (18.43%) while still roughly halving its drawdown. See `strategies/illiquidity_tilt.py`'s "Why a stop-loss" docstring section for the full sweep. |
| `illiquidity_tilt` (no stop-loss, superseded) | 17.04% | 1.01 | 69.1% | 24.03% | 110 | 2013-01-02 → 2026-09-25 | Underlying `amihud_illiquidity` signal confirmed weaker on Nifty 500 at the screening stage — see the Nifty 500 rows above for what actually happened when run there. Benchmark: equal-weight Nifty 50 buy-and-hold over the same window scored CAGR 18.43%, Sharpe 0.70, max drawdown 41.12%. Not yet written to `backtest_results` (run via research scripts, not the production pipeline). |

## No Nifty 500 run on record

Stored runs exist but are not a clean full-Nifty-500, full-history result —
listed here so a stale or partial number never gets mistaken for one.

| Strategy | Best number on record | Universe actually used | Why it's not usable as a Nifty 500 number |
|---|---|---|---|
| `rsi_mean_reversion` | CAGR 1.95%, Sharpe -0.46, win 64.1%, max dd 13.03%, 576 trades | 50 symbols, 2021-09-09 → 2026-09-07 | Predates the full-history/Nifty 500 extension — a 5-year, 50-symbol test run, not a full backtest. |
| `bollinger_breakout` | CAGR -2.16%, Sharpe -1.05, win 32.2%, max dd 18.74%, 603 trades | 50 symbols, 2021-09-09 → 2026-09-07 | Same situation as `rsi_mean_reversion`. |
| `sma_crossover` | — | Varies, usually <50 symbols across 506 stored runs | Almost all runs are small randomized symbol subsets (an old bootstrap/robustness sweep) — no single run represents "the" Nifty 500 result. |

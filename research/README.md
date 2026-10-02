# `research/` — Signal Screening Layer

Lighter and faster than the `strategies/` + `backtest.py` pipeline. Its job is to
answer one question cheaply, before you invest in building a full `Strategy`
class: **does this candidate signal have any statistical relationship with
future returns at all?** Nothing here writes to `signals` or any
`backtest_*` table — it only reads market data and writes to its own
`forward_returns` table.

Workflow: `research/screen.py` (screen candidates) → if a signal looks
promising → build it as a real `strategies.Strategy` → `backtest.py` (full
trade simulation with costs/slippage/sizing). See the root `README.md` for
`backtest_cli.py`'s manual (the CLI for that last step).

---

## Module-by-module

### `forward_returns.py` — the "ground truth" to test signals against

Computes **realized forward returns**: "if I'd bought at today's close, what
would I have made N trading days later?" This is the target variable every
signal gets correlated against.

- **`forward_returns` table** — one row per `(symbol, date)`, columns
  `fwd_return_1d`, `fwd_return_5d`, `fwd_return_10d`, `fwd_return_20d`, `fwd_return_40d`, `fwd_return_60d`.
- **`compute_forward_returns(df, horizons=[1,5,10,20])`** — per symbol, sorted
  by date, `fwd_return_Nd[T] = adj_close[T+N] / adj_close[T] - 1`, where
  `T+N` means N *trading days* ahead (via `.shift(-N)`), never N calendar
  days — skips weekends/holidays correctly. The last N rows per symbol are
  `NaN` (no future data yet) and are kept, not dropped.
- **`store_forward_returns(conn, df)`** — upsert into `forward_returns`
  (same insert-on-conflict-update pattern as `indicators.store_indicators`).
- **`compute_and_store_forward_returns(conn, symbols=None, horizons=None)`**
  — loads `adj_close` from `ohlcv_data` (the project's real daily-OHLCV
  table), computes, stores. Defaults to the active universe. Safe to re-run
  any time — it's an idempotent upsert.

You generally never call these directly — `screen.py` calls
`compute_and_store_forward_returns` automatically at the start of every run,
so the cache is always fresh.

### `signal_library.py` — the candidate signals themselves

A registry of small, self-contained functions, each computing one raw
numeric value per `(symbol, date)` row — **not** trading signals, just
values to be tested for predictive power. Each signal computes its own
indicator internally (never reads `indicators_daily`), so it can be
evaluated at *any* parameter value, not just whatever one default happens to
be pre-computed and stored.

| name | params (defaults) | what it measures | hypothesis |
|---|---|---|---|
| `momentum` | `window=20` | trailing N-day return | underreaction to news — recent winners keep winning for a while |
| `volume_weighted_momentum` | `window=20` | momentum × (volume ÷ its N-day average) | a price move on high volume reflects real conviction, not noise |
| `rsi_level` | `period=14` | raw RSI (0–100), not a crossing event | tests if the *level* alone has predictive power, before committing to a threshold rule |
| `bb_position` | `window=20, num_std=2.0` | `(price − lower band) / (upper − lower)`, 0=lower band, 1=upper band | tests whether "how stretched is price" predicts reversal or continuation — lets the data decide which |
| `volatility` | `window=20` | rolling N-day std of daily returns | could be a risk premium (positive) or distress signal (negative) — the screen tells you which dominates |
| `cross_sectional_rank_momentum` | `window=20` | same as `momentum`, but expressed as each stock's 0–1 percentile rank *among all symbols on that date* | strips out market-wide moves, tests purely relative attractiveness |
| `amihud_illiquidity` | `window=20` | rolling N-day mean of `\|daily adj_close return\| / (close × volume)` — price move per rupee actually traded | the liquidity premium: investors demand extra return for harder-to-trade stocks. Structurally different from every signal above — never looks at price direction/level at all, only price-impact-per-rupee-traded |

Look up a signal with `get_signal("momentum")`, which returns a `SignalSpec`
you call as `spec(df, params)`. List all registered names with
`available_signals()`.

**Methodology note:** with Spearman IC (the default), `momentum` and
`cross_sectional_rank_momentum` will *always* produce identical IC
statistics — percentile-ranking within a date is a monotonic transform, and
rank correlation is invariant to monotonic transforms. They only diverge
under `--method pearson`.

### `ic_analysis.py` — Information Coefficient: is there a relationship at all?

- **`calculate_ic(signal, fwd_return, dates, symbols, method='spearman')`**
  — for each date, correlates the signal value across every symbol with its
  forward return that same date (cross-sectional, not a time series for one
  stock). Returns one row per date: `date`, `ic`, `n_stocks`. `spearman`
  (rank correlation, the default) is computed as Pearson-correlation-of-ranks
  rather than via pandas' `method='spearman'`, since that calls
  `scipy.stats.spearmanr` — and `scipy` isn't installed in this project.
- **`summarize_ic(ic_df)`** — collapses the daily IC series into:
  - `mean_ic`, `std_ic`
  - `ic_ir` — `mean_ic / std_ic`, annualized by `√252` (same convention as
    `backtest.calculate_metrics`'s Sharpe ratio)
  - `pct_positive_days`
  - `t_stat` — one-sample t-test of whether mean IC differs from zero
  - `n_days`

  **How to read the numbers** (equity research rules of thumb, not law):
  | `|mean_ic|` | reading |
  |---|---|
  | ~0.02–0.05 | weak but *potentially real* — this is normal for a simple signal, not a disappointment |
  | > 0.05 | stronger by equity standards, worth real attention |
  | near 0, or `ic_ir`/`t_stat` not significant (`\|t_stat\| < 2`) | no real edge |
  | > 0.1 | **don't expect this** — rare even for well-known signals; usually a short/lucky sample or a data leak (look-ahead bias), not a discovery |

- **`plot_ic_over_time(ic_df, output_path)`** — bar chart of daily IC plus a
  60-day rolling mean overlay, saved to a PNG (headless `matplotlib`, no
  display needed).

### `decile_analysis.py` — what does the relationship actually look like?

IC gives you one number; this shows the *shape*. A signal can have a
decent-looking IC driven entirely by one extreme bucket — decile analysis
catches that.

- **`bucket_by_decile(signal, fwd_return, dates, symbols, n_buckets=5)`** —
  per date, ranks stocks into `n_buckets` equal-*count* groups by signal
  value (`pd.qcut`, `duplicates='drop'` so ties don't crash it), averages
  forward return per bucket per date.
- **`summarize_deciles(bucket_df)`** — averages across all dates: one row
  per bucket (1 = lowest signal value, N = highest) plus a `'spread'` row
  (top bucket mean − bottom bucket mean). **A real signal should show a
  roughly monotonic climb/fall across buckets — a flat or non-monotonic
  pattern means no meaningful signal, regardless of what the IC says.**
- **`plot_decile_returns(summary_df, output_path)`** — bar chart, bucket on
  x-axis, mean forward return on y-axis, saved to a PNG.

### `screen.py` — ties it all together into a CLI

See the manual below.

---

## CLI Manual: `python -m research.screen`

Run from the project root (`/home/krypton/Documents/workplace/trading_bot`).

```
python -m research.screen --help
python -m research.screen run --help
python -m research.screen batch --help
```

### Command: `run` — screen one signal

```
python -m research.screen run --signal SIGNAL [--params "k=v,k2=v2"]
    [--horizon {1d,5d,10d,20d,40d,60d}] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    [--universe nifty50|SYM1,SYM2,...] [--output-dir DIR]
    [--method {spearman,pearson}]
```

| flag | required | default | meaning |
|---|---|---|---|
| `--signal` | yes | — | signal name from the registry (see table above) |
| `--params` | no | signal's own defaults | `"key=value,key2=value2"`, e.g. `"window=10"` |
| `--horizon` | no | `5d` | which `forward_returns` column to test against |
| `--start` / `--end` | no | full available data range | screening window (`YYYY-MM-DD`) |
| `--universe` | no | `nifty50` | `nifty50` = active universe, or a comma-separated symbol list |
| `--output-dir` | no | `results/screens` | where plots + results JSON land |
| `--method` | no | `spearman` | IC correlation method |

**Examples:**

```bash
# Default momentum (20-day window), 5-day forward returns, full Nifty 50, full history
python -m research.screen run --signal momentum

# Same signal, custom window, explicit date range
python -m research.screen run --signal momentum --params window=10 \
    --start 2022-01-01 --end 2024-12-31

# RSI level at its default 14-day period, tested against 20-day forward returns
python -m research.screen run --signal rsi_level --horizon 20d

# Bollinger band position with a tighter band, on a hand-picked watchlist
python -m research.screen run --signal bb_position --params "window=15,num_std=1.5" \
    --universe RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK

# Use Pearson instead of Spearman (matters for cross_sectional_rank_momentum, see note above)
python -m research.screen run --signal cross_sectional_rank_momentum --method pearson
```

Each `run` prints a console summary (mean IC, std IC, annualized IC IR, %
positive days, t-stat, decile spread, plain-English verdict) and saves three
files to `--output-dir`:
- `{signal}_{horizon}_ic.png` — daily IC plot
- `{signal}_{horizon}_deciles.png` — bucket bar chart
- `{signal}_{horizon}_{timestamp}.json` — every number above, plus params and decile detail, for later comparison

### Command: `batch` — screen several signals and compare them

```
python -m research.screen batch --signals SIG1,SIG2,...
    [--horizon {1d,5d,10d,20d,40d,60d}] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
    [--universe nifty50|SYM1,SYM2,...] [--output-dir DIR]
    [--method {spearman,pearson}]
```

Same common flags as `run`, minus `--signal`/`--params` (each signal always
uses its own defaults in batch mode — use `run` if you need custom params).

**Examples:**

```bash
# All six signals, default 5-day horizon, full Nifty 50
python -m research.screen batch \
    --signals momentum,volume_weighted_momentum,rsi_level,bb_position,volatility,cross_sectional_rank_momentum

# Same, but 20-day horizon (run it again with a different --horizon to compare)
python -m research.screen batch \
    --signals momentum,volume_weighted_momentum,rsi_level,bb_position,volatility,cross_sectional_rank_momentum \
    --horizon 20d

# Just the momentum family, narrower date range
python -m research.screen batch --signals momentum,volume_weighted_momentum,cross_sectional_rank_momentum \
    --start 2023-01-01 --end 2025-12-31
```

`batch` loads OHLCV once and reuses it across every signal (not reloaded
per-signal), prints each signal's full individual summary as it runs, then
prints and saves one **comparison table sorted by `ic_ir` descending**:

```
      signal  mean_ic   ic_ir  t_stat  decile_spread                       verdict
  volatility   0.0313   2.591   5.649         0.0063   Weak but potentially real signal...
   rsi_level  -0.0074  -0.621  -1.356        -0.0011   No meaningful edge detected.
   ...
```

saved as `batch_{horizon}_{timestamp}.csv` in `--output-dir`.

**Note on sort order:** "descending" is literal numeric sort — a strongly
*negative* `ic_ir` sorts to the bottom even though a strong, reliable
negative signal (inverse predictor) is arguably just as useful as a strong
positive one. Check the full table, not just the top row, if a negative
result catches your eye.

### Tips

- `--horizon` only takes one value per invocation — to compare horizons
  (as in the examples above), just run `batch` twice with different
  `--horizon` values.
- Forward returns are computed automatically on first use for whatever
  universe you pass and cached in the `forward_returns` table; later runs
  reuse the cache (it's an idempotent upsert, always refreshed, so no stale
  data risk).
- `--universe` accepts raw tickers with or without `.NS` — both
  `RELIANCE` and `RELIANCE.NS` work.
- If `--signal` (or a name in `--signals`) isn't recognized, the error
  message lists every registered signal name.

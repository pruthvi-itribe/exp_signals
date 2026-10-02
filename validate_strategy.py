"""Parameter sensitivity search + train/test split, generic across strategies.

Why a train/test split at all: a parameter grid will always produce a "best"
row, even when the strategy has no real edge. The more combinations you test,
the more chances pure noise has to look like an edge on any *fixed* sample —
this is the same multiple-comparisons trap as p-hacking. Picking parameters
by their in-sample Sharpe ratio and then reporting that same in-sample Sharpe
as "the strategy's performance" is fooling yourself: you are grading a model
on the data it was chosen to look best on.

The fix is to never let parameter selection see the data you'll use to judge
it. This module produces an in-sample window for grid search
(``run_parameter_grid``) and a disjoint, later out-of-sample window that the
grid search never touches. You look at the in-sample grid, pick parameters
*by hand*, and only then run ``run_out_of_sample_test`` once on the held-out
window. A real edge degrades gracefully out-of-sample; a fitted-to-noise
"edge" tends to collapse or reverse sign. Running the out-of-sample test more
than once with different parameter choices reintroduces the same trap —
treat it as a single, final check, not another tuning round.

Grid-search combos are throwaway research state: signals are generated in
memory via ``strategies.registry.get_strategy(strategy_name)`` and never
written to the ``signals`` table, and backtests never write to any
``backtest_*`` table. Only a final, manually chosen out-of-sample run is
meant to represent something you'd act on.

Everything here is strategy-agnostic: it works with any class registered in
``strategies.registry``, driven entirely by ``strategy_name`` and a
``param_grid`` of config keyword-argument dicts — there is no SMA-specific
(or any other strategy-specific) logic in this file. A strategy declares what
raw columns it needs via ``required_columns``; this module loads a superset
(full daily OHLCV) and lets the strategy compute whatever indicators it needs
internally, at whatever parameters its config specifies.
"""

from __future__ import annotations

import duckdb
import pandas as pd

import backtest as bt
from strategies.registry import get_strategy
from src.market_regime import attach_market_regime, load_market_regime
from src.universe import DEFAULT_DB_PATH, get_active_universe

# Backtest engine settings applied identically to every grid combo and to the
# out-of-sample run, so results are comparable. Matches ``backtest.run_backtest``'s
# own defaults.
INITIAL_CAPITAL: float = 1_000_000
SLIPPAGE_PCT: float = 0.05
MAX_CONCURRENT_POSITIONS: int = 10

OVERFIT_WARNING: str = (
    "Do not just pick the single best row — check if performance is reasonably "
    "stable across 2-3 nearby parameter combos. A single standout result with "
    "neighbors performing much worse is a sign of overfitting to noise, not a "
    "real edge."
)

_OHLCV_COLUMNS: tuple[str, ...] = ("symbol", "date", "open", "high", "low", "close", "adj_close", "volume")


def _active_storage_symbols(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return active-universe symbols without the ``.NS`` suffix."""
    return [ticker.removesuffix(".NS") for ticker in get_active_universe(conn)]


def compute_in_sample_split(
    conn: duckdb.DuckDBPyConnection,
    in_sample_fraction: float = 0.8,
) -> tuple[str, str, str, str]:
    """Split the full available OHLCV history into in-sample / out-of-sample windows.

    The split point is computed from the actual distinct trading days present
    in ``ohlcv_data`` (not calendar days, which would be skewed by weekends),
    so ``in_sample_fraction`` really is that fraction of trading days.

    Args:
        conn: Open DuckDB connection.
        in_sample_fraction: Fraction of available trading days assigned to
            the in-sample window; the remainder is out-of-sample. This is the
            single knob for the split point — change the argument, not the
            body, to move it (e.g. 0.7 for a 70/30 split).

    Returns:
        ``(full_start, in_sample_end, out_of_sample_start, full_end)`` as ISO
        date strings. ``out_of_sample_start`` is the next actual trading day
        after ``in_sample_end``, so the two windows never overlap.
    """
    if not 0.0 < in_sample_fraction < 1.0:
        raise ValueError(f"in_sample_fraction must be between 0 and 1, got {in_sample_fraction}")

    trading_days = conn.execute(
        "SELECT DISTINCT timestamp::DATE AS date FROM ohlcv_data WHERE timeframe = '1d' ORDER BY date"
    ).df()["date"]

    if trading_days.empty:
        raise RuntimeError("No OHLCV data available to compute an in-sample/out-of-sample split.")

    split_idx = int(len(trading_days) * in_sample_fraction)
    split_idx = min(max(split_idx, 1), len(trading_days) - 1)  # leave >=1 day on each side

    full_start = trading_days.iloc[0].strftime("%Y-%m-%d")
    in_sample_end = trading_days.iloc[split_idx - 1].strftime("%Y-%m-%d")
    out_of_sample_start = trading_days.iloc[split_idx].strftime("%Y-%m-%d")
    full_end = trading_days.iloc[-1].strftime("%Y-%m-%d")

    return full_start, in_sample_end, out_of_sample_start, full_end


def _load_ohlcv_history(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    end_date: str,
) -> pd.DataFrame:
    """Load every available daily OHLCV bar up to ``end_date`` (no lower bound).

    A strategy's internal indicator computation needs history *before* the
    test window to be fully warmed up by the window's first day, so this has
    no lower bound — mirrors how ``indicators_daily`` is itself computed over
    full history rather than a date-bounded slice, so a grid combo isn't
    penalized with extra NaN warm-up rows purely because of where the test
    window happens to start.

    Loads the full OHLCV column set rather than just ``adj_close`` because
    this function is strategy-agnostic — different strategies declare
    different ``required_columns``, and this is a superset any of them can
    draw from. Also attaches the Nifty 50 market-regime columns (see
    ``src.market_regime``), same as ``src.strategy.load_strategy_input``, so
    a strategy that uses them behaves identically whether it's driven
    through the production pipeline or through this research tool.
    """
    if not symbols:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))

    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, timestamp::DATE AS date, open, high, low, close, adj_close, volume
        FROM ohlcv_data
        WHERE timeframe = '1d'
          AND symbol IN ({placeholders})
          AND timestamp::DATE <= ?
        ORDER BY symbol, date
    """
    params: list[object] = [*symbols, end_date]
    history = conn.execute(query, params).df()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    return attach_market_regime(history, load_market_regime(conn))


def _filter_to_range(signals_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Keep only signal rows whose ``date`` falls within ``[start_date, end_date]``."""
    if signals_df.empty:
        return signals_df
    in_range = signals_df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return signals_df.loc[in_range]


def _backtest_signals(signals_df: pd.DataFrame, price_df: pd.DataFrame) -> dict[str, object]:
    """Run the shared event-driven backtest engine on in-memory signals and return metrics.

    Reuses ``backtest``'s scheduling/simulation/metrics helpers exactly as
    ``backtest.run_backtest`` does, but skips ``_load_signals`` (signals come
    from a strategy's ``generate_signals`` here, not the ``signals`` table)
    and ``store_backtest_results`` (nothing here is a production backtest
    run worth persisting — this is a research tool).
    """
    if signals_df.empty or price_df.empty:
        trades_df = pd.DataFrame()
        equity_curve = pd.DataFrame()
    else:
        price_index = bt._build_price_index(price_df)
        close_matrix = (
            price_df.pivot(index="date", columns="symbol", values="close").sort_index().ffill()
        )
        scheduled = bt._schedule_executions(signals_df, price_index)
        trades, equity_rows, _skipped = bt._simulate(
            scheduled=scheduled,
            price_index=price_index,
            close_matrix=close_matrix,
            initial_capital=INITIAL_CAPITAL,
            slippage_pct=SLIPPAGE_PCT,
            max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
        )
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.DataFrame(equity_rows)
        if not equity_curve.empty:
            equity_curve = equity_curve.set_index("date")

    return bt.calculate_metrics(trades_df, equity_curve, INITIAL_CAPITAL)


def run_parameter_grid(
    conn: duckdb.DuckDBPyConnection,
    strategy_name: str,
    param_grid: list[dict],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Backtest each parameter combo in ``param_grid`` for one strategy, over one date range.

    Intended for the in-sample window only — pass the out-of-sample window
    here and any "winner" you pick is meaningless, since you'd have used the
    held-out data for selection (see the module docstring).

    For each dict of config kwargs in ``param_grid``: instantiates
    ``get_strategy(strategy_name)(**params)``, calls its
    ``generate_signals`` on the full loaded OHLCV history (in memory —
    nothing is written to the ``signals`` table), keeps only signals dated
    within ``[start_date, end_date]``, and backtests them with the shared
    engine (``_backtest_signals``, nothing written to any ``backtest_*``
    table).

    Args:
        conn: Open DuckDB connection.
        strategy_name: Registry key, e.g. ``"sma_crossover"`` — see
            ``strategies.registry.available_strategies()``.
        param_grid: Config keyword-argument dicts to test, e.g. for
            ``sma_crossover``:
            ``[{"fast_window": 10, "slow_window": 30}, {"fast_window": 20, "slow_window": 50}]``.
            Fields are whatever that strategy's ``StrategyConfig`` accepts —
            this function never inspects them.
        start_date: Inclusive lower bound of the test window (``YYYY-MM-DD``).
        end_date: Inclusive upper bound of the test window (``YYYY-MM-DD``).

    Returns:
        One row per parameter combo: every key from that combo's dict, plus
        ``cagr``, ``sharpe_ratio``, ``max_drawdown_pct``, ``total_trades``,
        ``win_rate``.
    """
    strategy_cls = get_strategy(strategy_name)
    symbols = _active_storage_symbols(conn)
    history = _load_ohlcv_history(conn, symbols, end_date)
    price_df = bt._load_daily_prices(conn, symbols, start_date, end_date)

    rows: list[dict[str, object]] = []
    for params in param_grid:
        strategy = strategy_cls(**params)
        signals_df = _filter_to_range(strategy.generate_signals(history), start_date, end_date)
        metrics = _backtest_signals(signals_df, price_df)

        rows.append(
            {
                **params,
                "cagr": metrics["cagr"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "total_trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
            }
        )

    return pd.DataFrame(rows)


def print_grid_results(results: pd.DataFrame) -> None:
    """Print the parameter grid sorted by Sharpe ratio, with an explicit overfitting caution."""
    print("\n=== In-sample parameter grid results (sorted by Sharpe ratio, descending) ===")
    if results.empty:
        print("No parameter grid results to display.")
        return

    ordered = results.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    print(
        ordered.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}" if pd.notna(value) else "NaN",
        )
    )
    print(f"\nNOTE: {OVERFIT_WARNING}")


def run_out_of_sample_test(
    conn: duckdb.DuckDBPyConnection,
    strategy_name: str,
    out_of_sample_start: str,
    out_of_sample_end: str,
    **strategy_params: object,
) -> dict[str, object]:
    """Run one manually chosen strategy configuration on held-out out-of-sample data.

    This is the real test. ``strategy_params`` should come from a human
    reading the ``run_parameter_grid`` output on in-sample data and picking a
    combo that looks good *and* has reasonable neighbors — never from
    searching this out-of-sample range for the best result, which would just
    move the overfitting problem here instead of solving it.

    Performance dropping somewhat versus the in-sample grid is normal and
    expected (the in-sample number is usually a mild overestimate even for a
    real edge). Performance collapsing to roughly zero, or Sharpe/CAGR
    flipping sign, is a red flag that the in-sample result was noise rather
    than a real, persistent edge.

    Args:
        conn: Open DuckDB connection.
        strategy_name: Registry key, e.g. ``"sma_crossover"``.
        out_of_sample_start: Inclusive lower bound (``YYYY-MM-DD``).
        out_of_sample_end: Inclusive upper bound (``YYYY-MM-DD``).
        **strategy_params: Config kwargs for this strategy, manually chosen
            from in-sample results (e.g. ``fast_window=20, slow_window=50``
            for ``sma_crossover``).

    Returns:
        The ``calculate_metrics`` summary dict for this configuration on the
        out-of-sample window (``total_trades``, ``win_rate``,
        ``total_return_pct``, ``cagr``, ``max_drawdown_pct``,
        ``sharpe_ratio``, ``final_equity``).
    """
    strategy = get_strategy(strategy_name)(**strategy_params)

    symbols = _active_storage_symbols(conn)
    history = _load_ohlcv_history(conn, symbols, out_of_sample_end)
    price_df = bt._load_daily_prices(conn, symbols, out_of_sample_start, out_of_sample_end)

    signals_df = _filter_to_range(
        strategy.generate_signals(history), out_of_sample_start, out_of_sample_end
    )
    return _backtest_signals(signals_df, price_df)


if __name__ == "__main__":
    STRATEGY_NAME = "sma_crossover"
    IN_SAMPLE_FRACTION = 0.8

    PARAM_GRID: list[dict] = [
        {"fast_window": 10, "slow_window": 30},
        {"fast_window": 15, "slow_window": 40},
        {"fast_window": 20, "slow_window": 50},
        {"fast_window": 30, "slow_window": 100},
    ]

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        full_start, in_sample_end, out_of_sample_start, full_end = compute_in_sample_split(
            conn, in_sample_fraction=IN_SAMPLE_FRACTION
        )

        print(f"Strategy: {STRATEGY_NAME}")
        print(f"Full available range: {full_start} to {full_end}")
        print(
            f"In-sample:     {full_start} to {in_sample_end}  "
            f"({IN_SAMPLE_FRACTION:.0%} of trading days)"
        )
        print(
            f"Out-of-sample: {out_of_sample_start} to {full_end}  "
            f"({1 - IN_SAMPLE_FRACTION:.0%} of trading days, held out — not used below)"
        )

        grid_results = run_parameter_grid(
            conn,
            strategy_name=STRATEGY_NAME,
            param_grid=PARAM_GRID,
            start_date=full_start,
            end_date=in_sample_end,
        )
        print_grid_results(grid_results)

        # --- Manual step ---
        # Review the printed grid above yourself. Pick a parameter combo with a
        # good Sharpe ratio AND reasonable 2-3 nearest neighbors — not just the
        # single best row (see the caution printed above). Then fill in your
        # choice below and uncomment this block to run the one-shot, real
        # out-of-sample check:
        #
        # chosen_params = {"fast_window": 20, "slow_window": 50}  # <- replace with your pick
        # oos_metrics = run_out_of_sample_test(
        #     conn,
        #     strategy_name=STRATEGY_NAME,
        #     out_of_sample_start=out_of_sample_start,
        #     out_of_sample_end=full_end,
        #     **chosen_params,
        # )
        # print(f"\n=== Out-of-sample test: {STRATEGY_NAME} {chosen_params} ===")
        # print(oos_metrics)
    finally:
        conn.close()

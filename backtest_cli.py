"""Interactive CLI for running and inspecting strategies/ backtests.

Lets you list what's registered, see each strategy's tunable parameters,
run a backtest (with parameter overrides) against a single symbol or the
full active universe, and revisit or compare stored results — without
writing a one-off script each time. Mirrors ``research/screen.py``'s style
and structure (same ``argparse`` subcommand pattern, same
load/compute/print/save shape) for consistency across the project's CLIs.

Usage::

    python -m backtest_cli list-strategies
    python -m backtest_cli list-params --strategy sma_crossover
    python -m backtest_cli run --strategy sma_crossover --params fast_window=10,slow_window=30 \\
        --symbol RELIANCE --start 2023-01-01 --end 2024-12-31
    python -m backtest_cli show-results --latest --strategy sma_crossover_10_30
    python -m backtest_cli compare --strategy sma_crossover_20_50,rsi_mean_reversion

Important: ``run_backtest()`` in ``backtest.py`` does NOT itself generate
signals — it only backtests signals already sitting in the ``signals``
table. Generating them is a separate step, ``src.strategy.run_strategy()``.
This CLI calls both, in sequence, exactly as ``extend_history.py`` already
does — it does not duplicate either one.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from backtest import run_backtest
from src.strategy import run_strategy
from src.universe import DEFAULT_DB_PATH, get_active_universe
from strategies.base import Strategy
from strategies.registry import available_strategies, get_strategy

DAILY_TIMEFRAME: str = "1d"
DEFAULT_OUTPUT_DIR: str = "results/backtests"

METRIC_KEYS: tuple[str, ...] = (
    "total_trades",
    "win_rate",
    "total_return_pct",
    "cagr",
    "max_drawdown_pct",
    "sharpe_ratio",
    "final_equity",
)


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def _first_doc_line(cls: type) -> str:
    """Return the first line of a class's docstring, or a placeholder if it has none."""
    doc = inspect.getdoc(cls)
    if not doc:
        return "(no description)"
    return doc.strip().splitlines()[0]


def _strategy_not_found_message(name: str) -> str:
    return (
        f"Unknown strategy '{name}'. Available: {', '.join(available_strategies())}. "
        f"Run 'python -m backtest_cli list-strategies' for details."
    )


def _parse_params(params_str: str | None) -> dict[str, object]:
    """Parse a ``"key=value,key2=value2"`` CLI string into a typed dict.

    Each value is coerced to ``int``, then ``float``, falling back to the
    raw string (e.g. for a string-valued param like ``exit_mode``). Mirrors
    ``research/screen.py``'s ``_parse_params`` — reimplemented locally
    rather than imported, since ``backtest_cli.py`` and ``research/`` are
    intentionally independent CLI tools (see ``ARCHITECTURE.md``).
    """
    if not params_str:
        return {}

    result: dict[str, object] = {}
    for pair in params_str.split(","):
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        try:
            result[key] = int(raw_value)
        except ValueError:
            try:
                result[key] = float(raw_value)
            except ValueError:
                result[key] = raw_value
    return result


def _build_strategy(strategy_name: str, params: dict[str, object]) -> Strategy | None:
    """Look up, validate params for, and construct a strategy. Prints its own errors.

    Returns ``None`` (after printing a clear message to stderr) if the
    strategy name is unknown, a param name isn't recognized, or the
    resulting config fails its ``validate()`` check — so callers can just
    check for ``None`` and return exit code 1.
    """
    if strategy_name not in available_strategies():
        print(f"Error: {_strategy_not_found_message(strategy_name)}", file=sys.stderr)
        return None

    strategy_cls = get_strategy(strategy_name)
    param_info = strategy_cls.config_cls.param_info()
    unknown = sorted(set(params) - set(param_info))
    if unknown:
        valid = ", ".join(sorted(param_info)) or "(none)"
        print(
            f"Error: unknown parameter(s) for '{strategy_name}': {', '.join(unknown)}. "
            f"Valid parameters: {valid}. "
            f"Run 'python -m backtest_cli list-params --strategy {strategy_name}' for details.",
            file=sys.stderr,
        )
        return None

    try:
        strategy = strategy_cls(**params)
    except (ValueError, TypeError) as exc:
        print(f"Error: invalid parameters for '{strategy_name}': {exc}", file=sys.stderr)
        return None

    if strategy.name == strategy.base_name:
        print(
            f"Note: '{strategy_name}' stores signals under the fixed name '{strategy.name}' "
            f"regardless of parameters — running it again later with *different* --params "
            f"will overwrite previously stored signals at overlapping symbol/date rows.",
        )

    return strategy


def _resolve_date_range(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: str | None,
    end: str | None,
) -> tuple[str, str]:
    """Resolve ``--start``/``--end``, defaulting to the full available OHLCV range."""
    if start and end:
        return start, end

    placeholders = ", ".join("?" for _ in symbols)
    row = conn.execute(
        f"""
        SELECT MIN(timestamp)::DATE, MAX(timestamp)::DATE
        FROM ohlcv_data
        WHERE timeframe = ? AND symbol IN ({placeholders})
        """,
        [DAILY_TIMEFRAME, *symbols],
    ).fetchone()
    if row[0] is None:
        raise RuntimeError("No OHLCV data available for the requested symbol(s).")
    return start or str(row[0]), end or str(row[1])


# --------------------------------------------------------------------------
# Loading / printing / saving a run's summary (shared by `run` and `show-results`)
# --------------------------------------------------------------------------


def _load_run_summary(conn: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, object] | None:
    """Reconstruct a run's summary purely from stored tables. Returns ``None`` if not found.

    Note on what's NOT recoverable from storage alone: ``backtest_runs``
    does not persist the symbol scope or individual strategy parameters —
    only the stored ``strategy_name``, which only encodes parameters for
    strategies whose ``name`` folds config into it (``sma_crossover``, e.g.
    ``sma_crossover_10_30``) and not for flat-named ones
    (``rsi_mean_reversion``, ``bollinger_breakout``). Symbol scope is
    approximated here from the distinct symbols in ``backtest_trades``,
    which is empty — and so unrecoverable — for a run with zero trades.
    """
    run_row = conn.execute(
        """
        SELECT run_id, strategy_name, start_date, end_date, initial_capital, position_sizing, created_at
        FROM backtest_runs WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    if run_row is None:
        return None

    result_row = conn.execute(
        f"SELECT {', '.join(METRIC_KEYS)} FROM backtest_results WHERE run_id = ?", [run_id]
    ).fetchone()
    metrics = dict(zip(METRIC_KEYS, result_row)) if result_row else None

    trades_df = conn.execute(
        """
        SELECT symbol, signal_date, entry_date, entry_price, quantity,
               exit_date, exit_price, exit_reason, gross_pnl, net_pnl
        FROM backtest_trades WHERE run_id = ?
        """,
        [run_id],
    ).df()
    symbols = sorted(trades_df["symbol"].unique().tolist()) if not trades_df.empty else []

    return {
        "run_id": run_row[0],
        "strategy_name": run_row[1],
        "start_date": str(run_row[2]),
        "end_date": str(run_row[3]),
        "initial_capital": run_row[4],
        "position_sizing": run_row[5],
        "created_at": str(run_row[6]),
        "symbols": symbols,
        "symbols_recoverable": bool(symbols),
        "metrics": metrics,
        "trades_df": trades_df,
    }


def _print_run_summary(result: dict[str, object], params: dict[str, object] | None) -> None:
    """Print the console summary shared by `run` (just-computed) and `show-results` (reloaded)."""
    print(f"\n=== Backtest run: {result['run_id']} ===")
    print(f"Strategy:        {result['strategy_name']}")
    if params is not None:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "(defaults)"
        print(f"Params:          {param_str}")
    else:
        print("Params:          not recoverable from stored data (see the saved JSON from the original run, if any)")
    print(f"Date range:      {result['start_date']} to {result['end_date']}")
    if result["symbols_recoverable"]:
        symbols = result["symbols"]
        scope = symbols[0] if len(symbols) == 1 else f"{len(symbols)} symbols"
        print(f"Symbol scope:    {scope}")
    else:
        print("Symbol scope:    not recoverable (zero trades in this run)")
    print(f"Initial capital: {result['initial_capital']:,.2f}   Position sizing: {result['position_sizing']}")

    metrics = result["metrics"]
    if metrics is None:
        print("\nNo stored metrics found for this run.")
        return

    print("\nMetrics:")
    print(f"  CAGR:             {metrics['cagr']:.3f}%")
    print(f"  Sharpe ratio:     {metrics['sharpe_ratio']:.3f}")
    print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.3f}%")
    print(f"  Win rate:         {metrics['win_rate']:.1%}")
    print(f"  Total trades:     {int(metrics['total_trades'])}")
    print(f"  Final equity:     {metrics['final_equity']:,.2f}")

    trades_df = result["trades_df"]
    if trades_df.empty:
        print("\nNo trades were closed in this run.")
        return

    print("\nTop 5 trades by net PnL:")
    print(trades_df.nlargest(5, "net_pnl").to_string(index=False))
    print("\nBottom 5 trades by net PnL:")
    print(trades_df.nsmallest(5, "net_pnl").to_string(index=False))


def _save_run_summary(
    result: dict[str, object],
    params: dict[str, object] | None,
    output_dir: Path,
) -> Path:
    """Save the run summary as a timestamped JSON file, independent of the DB.

    Unlike ``backtest_runs``, this file always records the full resolved
    params (defaults + overrides) and the exact symbol scope, since those
    aren't fully recoverable from storage alone for every strategy (see
    ``_load_run_summary``'s docstring) — this JSON is the durable record of
    exactly what a given ``run_id`` was.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{result['strategy_name']}_{result['run_id'][:8]}_{timestamp}.json"

    trades_df: pd.DataFrame = result["trades_df"]
    payload = {
        "run_id": result["run_id"],
        "strategy_name": result["strategy_name"],
        "params": params,
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "symbols": result["symbols"],
        "initial_capital": result["initial_capital"],
        "position_sizing": result["position_sizing"],
        "metrics": result["metrics"],
        "trades": trades_df.to_dict(orient="records"),
        "generated_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def _list_strategies_command(args: argparse.Namespace) -> int:
    """`list-strategies`: name, one-line description, required columns, per strategy."""
    names = available_strategies()
    if not names:
        print("No strategies registered.")
        return 0

    print(f"=== Registered strategies ({len(names)}) ===\n")
    for name in names:
        cls = get_strategy(name)
        print(f"{name}")
        print(f"  {_first_doc_line(cls)}")
        print(f"  required columns: {', '.join(cls.required_columns)}")
        print()
    return 0


def _list_params_command(args: argparse.Namespace) -> int:
    """`list-params --strategy NAME`: each tunable parameter, its default, and description."""
    if args.strategy not in available_strategies():
        print(f"Error: {_strategy_not_found_message(args.strategy)}", file=sys.stderr)
        return 1

    cls = get_strategy(args.strategy)
    param_info = cls.config_cls.param_info()

    print(f"=== Parameters for '{args.strategy}' ===\n")
    if not param_info:
        print("(no tunable parameters)")
        return 0

    for info in param_info.values():
        description = info.description or "(no description)"
        print(f"{info.name} (default={info.default!r})")
        print(f"  {description}")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    """`run`: build the strategy, generate + store signals, backtest, print + save results."""
    params = _parse_params(args.params)
    strategy = _build_strategy(args.strategy, params)
    if strategy is None:
        return 1

    resolved_params = asdict(strategy.config)

    index_name = args.universe.upper()  # "nifty50" -> "NIFTY50", "nifty500" -> "NIFTY500"

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        if args.symbol:
            symbol = args.symbol.strip().upper().removesuffix(".NS")
            active = sorted(t.removesuffix(".NS") for t in get_active_universe(conn, index_name=index_name))
            if symbol not in active:
                sample = ", ".join(active[:10])
                print(
                    f"Error: '{symbol}' is not in the active {index_name} universe "
                    f"({len(active)} symbols, no '.NS' suffix needed). "
                    f"Check the spelling, try --universe nifty500 if it's not a Nifty 50 "
                    f"constituent — e.g. {sample}, ... — or omit --symbol to "
                    f"backtest the full {index_name} universe instead.",
                    file=sys.stderr,
                )
                return 1
            symbols = [symbol]
        else:
            symbols = [t.removesuffix(".NS") for t in get_active_universe(conn, index_name=index_name)]

        try:
            start_date, end_date = _resolve_date_range(conn, symbols, args.start, args.end)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(
            f"Running '{strategy.name}' on {len(symbols)} symbol(s) "
            f"({symbols[0] if len(symbols) == 1 else 'full active universe'}) "
            f"| {start_date} to {end_date}"
        )

        # Signal generation is a separate step from backtesting (see module
        # docstring) — run_strategy() populates `signals`, run_backtest()
        # then reads from it.
        run_strategy(conn, strategy=strategy, symbols=symbols)

        run_id = run_backtest(
            conn,
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.initial_capital,
            position_sizing=args.position_sizing,
            slippage_pct=args.slippage_pct,
            symbols=symbols,
            max_concurrent_positions=args.max_positions,
        )

        result = _load_run_summary(conn, run_id)
    finally:
        conn.close()

    if result is None:
        print("Error: backtest completed but the run could not be reloaded from storage.", file=sys.stderr)
        return 1

    _print_run_summary(result, params=resolved_params)
    saved_path = _save_run_summary(result, resolved_params, Path(args.output_dir))
    print(f"\nSaved: {saved_path}")
    return 0


def _show_results_command(args: argparse.Namespace) -> int:
    """`show-results`: re-print a stored run's summary without re-running it."""
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        if args.latest:
            if not args.strategy:
                print("Error: --latest requires --strategy (the stored strategy_name to look up).", file=sys.stderr)
                return 1
            row = conn.execute(
                "SELECT run_id FROM backtest_runs WHERE strategy_name = ? ORDER BY created_at DESC LIMIT 1",
                [args.strategy],
            ).fetchone()
            if row is None:
                print(
                    f"Error: no stored backtest runs found for strategy_name '{args.strategy}'. "
                    f"Note this must be the exact stored name (e.g. 'sma_crossover_20_50'), not "
                    f"necessarily the registry key — run 'python -m backtest_cli compare --strategy "
                    f"{args.strategy}' or query backtest_runs directly to check what's stored.",
                    file=sys.stderr,
                )
                return 1
            run_id = row[0]
        else:
            run_id = args.run_id

        result = _load_run_summary(conn, run_id)
        if result is None:
            print(f"Error: no backtest run found with run_id '{run_id}'.", file=sys.stderr)
            return 1

        _print_run_summary(result, params=None)
    finally:
        conn.close()
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    """`compare`: side-by-side metrics table across multiple stored runs."""
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        if args.run_ids:
            run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
        else:
            names = [s.strip() for s in args.strategy.split(",") if s.strip()]
            run_ids = []
            for name in names:
                row = conn.execute(
                    "SELECT run_id FROM backtest_runs WHERE strategy_name = ? ORDER BY created_at DESC LIMIT 1",
                    [name],
                ).fetchone()
                if row is None:
                    print(f"Warning: no stored runs found for strategy_name '{name}', skipping.", file=sys.stderr)
                    continue
                run_ids.append(row[0])

        rows = []
        for run_id in run_ids:
            result = _load_run_summary(conn, run_id)
            if result is None or result["metrics"] is None:
                print(f"Warning: run_id '{run_id}' not found or has no stored metrics, skipping.", file=sys.stderr)
                continue
            m = result["metrics"]
            rows.append(
                {
                    "strategy_name": result["strategy_name"],
                    "run_id": result["run_id"][:8],
                    "cagr": m["cagr"],
                    "sharpe_ratio": m["sharpe_ratio"],
                    "max_drawdown_pct": m["max_drawdown_pct"],
                    "win_rate": m["win_rate"],
                    "total_trades": int(m["total_trades"]),
                }
            )
    finally:
        conn.close()

    if not rows:
        print("Error: nothing to compare — no matching runs with stored metrics were found.", file=sys.stderr)
        return 1

    table = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    print(f"\n=== Backtest comparison ({len(table)} runs, sorted by Sharpe ratio descending) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


# --------------------------------------------------------------------------
# argparse wiring
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backtest_cli",
        description=(
            "Run and inspect strategies/ backtests from the command line: list what's "
            "registered, see each strategy's tunable parameters, run a backtest (with "
            "parameter overrides, against a single symbol or the full active universe), "
            "and revisit or compare stored results — without writing a one-off script "
            "each time."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-strategies", help="List all registered strategies.")

    list_params_parser = subparsers.add_parser("list-params", help="List a strategy's tunable parameters.")
    list_params_parser.add_argument("--strategy", required=True, help="Registry name, e.g. sma_crossover.")

    run_parser = subparsers.add_parser("run", help="Run a backtest for one strategy.")
    run_parser.add_argument("--strategy", required=True, help="Registry name, e.g. sma_crossover.")
    run_parser.add_argument(
        "--params", default=None, help="key=value pairs, comma-separated, e.g. 'fast_window=10,slow_window=30'."
    )
    run_parser.add_argument("--symbol", default=None, help="Single symbol (e.g. RELIANCE). Default: full active universe.")
    run_parser.add_argument(
        "--universe",
        default="nifty50",
        choices=["nifty50", "nifty500"],
        help="Which tracked universe --symbol is validated against (and --symbol's default when omitted). Default: nifty50.",
    )
    run_parser.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD). Default: full available range.")
    run_parser.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD). Default: full available range.")
    run_parser.add_argument("--initial-capital", type=float, default=1_000_000, help="Starting capital. Default: 1000000.")
    run_parser.add_argument(
        "--position-sizing",
        default="equal_weight",
        choices=["equal_weight"],
        help="Position sizing scheme (only 'equal_weight' is currently implemented).",
    )
    run_parser.add_argument("--slippage-pct", type=float, default=0.05, help="Slippage in percentage points. Default: 0.05.")
    run_parser.add_argument("--max-positions", type=int, default=10, help="Max concurrent open positions. Default: 10.")
    run_parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Where to save the results JSON. Default: {DEFAULT_OUTPUT_DIR}."
    )

    show_parser = subparsers.add_parser("show-results", help="Re-print a stored run's summary without re-running it.")
    show_group = show_parser.add_mutually_exclusive_group(required=True)
    show_group.add_argument("--run-id", default=None, help="Exact run_id to show.")
    show_group.add_argument("--latest", action="store_true", help="Show the most recent stored run for --strategy.")
    show_parser.add_argument(
        "--strategy",
        default=None,
        help="Stored strategy_name (e.g. sma_crossover_20_50 — the exact backtest_runs.strategy_name value, "
        "not necessarily the registry key). Required with --latest.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare stored runs side by side.")
    compare_group = compare_parser.add_mutually_exclusive_group(required=True)
    compare_group.add_argument(
        "--strategy", default=None, help="Comma-separated stored strategy_name(s); compares each one's latest run."
    )
    compare_group.add_argument("--run-ids", default=None, help="Comma-separated run_id(s) to compare directly.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "list-strategies": _list_strategies_command,
        "list-params": _list_params_command,
        "run": _run_command,
        "show-results": _show_results_command,
        "compare": _compare_command,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

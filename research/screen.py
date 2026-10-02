"""CLI entry point for the signal-screening research layer.

Screens whether a candidate signal has statistical predictive power on
forward returns BEFORE building a full trading strategy around it — this is
a lighter, faster research step that sits ahead of ``strategies/`` +
``backtest.py``, meant to avoid sinking effort into a full ``Strategy`` class
and backtest for a signal that never had any real edge to begin with.

Usage::

    python -m research.screen run --signal momentum --params window=20 \\
        --horizon 5d --start 2020-01-01 --end 2024-12-31 \\
        --universe nifty50 --output-dir results/screens

    python -m research.screen batch --signals momentum,rsi_level,bb_position \\
        --horizon 5d --output-dir results/screens
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from research.decile_analysis import bucket_by_decile, plot_decile_returns, summarize_deciles
from research.forward_returns import compute_and_store_forward_returns, ensure_forward_returns_schema
from research.ic_analysis import calculate_ic, plot_ic_over_time, summarize_ic
from research.signal_library import available_signals, get_signal
from src.universe import DEFAULT_DB_PATH, get_active_universe

DAILY_TIMEFRAME: str = "1d"
HORIZON_CHOICES: tuple[str, ...] = ("1d", "5d", "10d", "20d")
DEFAULT_OUTPUT_DIR: str = "results/screens"

# Significance requires both a non-trivial sample and a t-stat past the
# conventional ~95% one-sample-t threshold; magnitude bands follow
# research.ic_analysis.summarize_ic's docstring.
MIN_DAYS_FOR_VERDICT: int = 20
T_STAT_SIGNIFICANCE: float = 2.0
WEAK_IC_THRESHOLD: float = 0.02
STRONG_IC_THRESHOLD: float = 0.05


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def _resolve_universe_symbols(conn: duckdb.DuckDBPyConnection, universe: str) -> list[str]:
    """Resolve ``--universe`` into a list of storage symbols (no ``.NS`` suffix).

    ``'nifty50'`` (case-insensitive) means the active universe; anything
    else is treated as a comma-separated custom symbol list.
    """
    if universe.strip().lower() == "nifty50":
        return [t.removesuffix(".NS") for t in get_active_universe(conn)]
    return [s.strip().removesuffix(".NS") for s in universe.split(",") if s.strip()]


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
        raise RuntimeError("No OHLCV data available for the requested universe.")
    return start or str(row[0]), end or str(row[1])


def _load_ohlcv_history(conn: duckdb.DuckDBPyConnection, symbols: list[str], end_date: str) -> pd.DataFrame:
    """Load every available daily OHLCV bar up to ``end_date`` (no lower bound).

    A signal's internal indicator computation needs history *before* the
    screening window to be warmed up by the window's first day — mirrors
    the same pattern ``validate_strategy.py`` uses for its parameter grid.
    """
    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, timestamp::DATE AS date, open, high, low, close, adj_close, volume
        FROM ohlcv_data
        WHERE timeframe = ?
          AND symbol IN ({placeholders})
          AND timestamp::DATE <= ?
        ORDER BY symbol, date
    """
    df = conn.execute(query, [DAILY_TIMEFRAME, *symbols, end_date]).df()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _ensure_forward_returns(conn: duckdb.DuckDBPyConnection, symbols: list[str]) -> None:
    """Compute and upsert forward returns for ``symbols`` if missing.

    Always (re)computes rather than trying to detect partial/stale
    coverage: it's a cheap, idempotent upsert over the whole universe's
    history, so recomputing is simpler and more robust than a fragile
    "is it populated enough" check, and it guarantees freshness if new
    OHLCV data has landed since the last screen.
    """
    ensure_forward_returns_schema(conn)
    compute_and_store_forward_returns(conn, symbols=symbols)


def _load_forward_returns(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start_date: str,
    end_date: str,
    horizon_column: str,
) -> pd.DataFrame:
    """Load one horizon's forward returns from ``forward_returns`` for the screening window."""
    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, date, {horizon_column} AS fwd_return
        FROM forward_returns
        WHERE symbol IN ({placeholders}) AND date BETWEEN ? AND ?
    """
    df = conn.execute(query, [*symbols, start_date, end_date]).df()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _parse_params(params_str: str | None) -> dict:
    """Parse a ``"key=value,key2=value2"`` CLI string into a typed dict.

    Each value is coerced to ``int``, then ``float``, falling back to the
    raw string (e.g. for a future string-valued param like ``exit_mode``).
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


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def verdict_for(mean_ic: float, t_stat: float, n_days: int) -> str:
    """Plain-English verdict from IC diagnostics, using conservative equity-research thresholds.

    See ``research.ic_analysis.summarize_ic`` for the reasoning behind the
    magnitude bands; this adds the sample-size and significance gate.
    """
    if n_days < MIN_DAYS_FOR_VERDICT or pd.isna(t_stat) or pd.isna(mean_ic):
        return "Insufficient data to draw a conclusion."
    if abs(t_stat) < T_STAT_SIGNIFICANCE or abs(mean_ic) < WEAK_IC_THRESHOLD:
        return "No meaningful edge detected."
    if abs(mean_ic) < STRONG_IC_THRESHOLD:
        return "Weak but potentially real signal — worth a closer look before building a strategy."
    return "Signal shows edge, worth building a strategy around."


# --------------------------------------------------------------------------
# Core screening
# --------------------------------------------------------------------------


def screen_signal(
    conn: duckdb.DuckDBPyConnection,
    signal_name: str,
    params: dict,
    horizon: str,
    start_date: str,
    end_date: str,
    symbols: list[str],
    method: str,
    output_dir: Path,
    history: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run the full screen for one signal: compute it, score it, plot it, save it.

    Args:
        conn: Open DuckDB connection.
        signal_name: Registry key from ``research.signal_library``.
        params: Config overrides for this signal (merged over its defaults).
        horizon: One of ``HORIZON_CHOICES``.
        start_date: Inclusive lower bound of the screening window (``YYYY-MM-DD``).
        end_date: Inclusive upper bound (``YYYY-MM-DD``).
        symbols: Universe to screen over.
        method: ``'spearman'`` or ``'pearson'``.
        output_dir: Where to save plots and the results file.
        history: Pre-loaded full OHLCV history (``_load_ohlcv_history``'s
            output), shared across signals in a batch run so it's loaded
            once rather than once per signal. Loaded here if omitted.

    Returns:
        A flat dict of everything needed for console printing, a batch
        comparison table, and the saved results file.
    """
    spec = get_signal(signal_name)
    merged_params = {**spec.default_params, **params}
    horizon_column = f"fwd_return_{horizon}"

    if history is None:
        history = _load_ohlcv_history(conn, symbols, end_date)

    signal_values = spec(history, merged_params)
    signal_df = history[["symbol", "date"]].copy()
    signal_df["signal"] = signal_values.values
    in_range = signal_df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    signal_df = signal_df.loc[in_range]

    forward_returns = _load_forward_returns(conn, symbols, start_date, end_date, horizon_column)
    merged = signal_df.merge(forward_returns, on=["symbol", "date"], how="inner")
    merged = merged.dropna(subset=["signal", "fwd_return"])

    ic_df = calculate_ic(merged["signal"], merged["fwd_return"], merged["date"], merged["symbol"], method=method)
    ic_summary = summarize_ic(ic_df)

    bucket_df = bucket_by_decile(merged["signal"], merged["fwd_return"], merged["date"], merged["symbol"])
    decile_summary = summarize_deciles(bucket_df)
    decile_spread = (
        float(decile_summary.loc["spread", "mean_fwd_return"]) if "spread" in decile_summary.index else float("nan")
    )

    verdict = verdict_for(ic_summary["mean_ic"], ic_summary["t_stat"], ic_summary["n_days"])

    output_dir.mkdir(parents=True, exist_ok=True)
    ic_plot_path = plot_ic_over_time(ic_df, output_dir / f"{signal_name}_{horizon}_ic.png")
    decile_plot_path = plot_decile_returns(decile_summary, output_dir / f"{signal_name}_{horizon}_deciles.png")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"{signal_name}_{horizon}_{timestamp}.json"
    payload = {
        "signal": signal_name,
        "params": merged_params,
        "horizon": horizon,
        "method": method,
        "start_date": start_date,
        "end_date": end_date,
        "n_symbols": len(symbols),
        "n_merged_rows": len(merged),
        "ic_summary": ic_summary,
        "decile_summary": decile_summary.reset_index().rename(columns={"index": "bucket"}).to_dict(orient="records"),
        "decile_spread": decile_spread,
        "verdict": verdict,
        "generated_at": datetime.now().isoformat(),
    }
    results_path.write_text(json.dumps(payload, indent=2, default=str))

    return {
        "signal": signal_name,
        "params": merged_params,
        "horizon": horizon,
        **ic_summary,
        "decile_spread": decile_spread,
        "verdict": verdict,
        "ic_plot_path": str(ic_plot_path),
        "decile_plot_path": str(decile_plot_path),
        "results_path": str(results_path),
    }


# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------


def _print_single_result(result: dict[str, object]) -> None:
    params_str = ", ".join(f"{k}={v}" for k, v in result["params"].items())
    print(
        f"\n=== Screen: {result['signal']} ({params_str}) | horizon={result['horizon']} "
        f"| n_days={result['n_days']} ==="
    )
    print(f"  Mean IC:          {result['mean_ic']:.4f}")
    print(f"  Std IC:           {result['std_ic']:.4f}")
    print(f"  IC IR (ann.):     {result['ic_ir']:.3f}")
    print(f"  % positive days:  {result['pct_positive_days']:.1%}")
    print(f"  t-stat:           {result['t_stat']:.3f}")
    print(f"  Decile spread:    {result['decile_spread']:.5f}")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Saved: {result['ic_plot_path']}")
    print(f"  Saved: {result['decile_plot_path']}")
    print(f"  Saved: {result['results_path']}")


def _comparison_table(results: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "signal": r["signal"],
                "mean_ic": r["mean_ic"],
                "ic_ir": r["ic_ir"],
                "t_stat": r["t_stat"],
                "decile_spread": r["decile_spread"],
                "verdict": r["verdict"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("ic_ir", ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m research.screen",
        description=(
            "Screen whether a candidate signal has statistical predictive power on "
            "forward returns BEFORE building a full trading strategy around it. "
            "This is a lighter, faster research layer than strategies/ + backtest.py, "
            "meant to avoid spending effort building and backtesting a strategy for a "
            "signal that never had any real underlying edge."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Screen a single candidate signal.")
    run_parser.add_argument(
        "--signal", required=True, help=f"Signal name from the registry. Available: {', '.join(available_signals())}"
    )
    run_parser.add_argument("--params", default=None, help="key=value pairs, comma-separated, e.g. 'window=20,num_std=2.0'.")
    _add_common_args(run_parser)

    batch_parser = subparsers.add_parser("batch", help="Screen multiple candidate signals and compare them.")
    batch_parser.add_argument("--signals", required=True, help="Comma-separated signal names from the registry.")
    _add_common_args(batch_parser)

    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", default="5d", choices=HORIZON_CHOICES, help="Forward-return horizon to test against.")
    p.add_argument("--start", default=None, help="Inclusive start date (YYYY-MM-DD). Default: full available range.")
    p.add_argument("--end", default=None, help="Inclusive end date (YYYY-MM-DD). Default: full available range.")
    p.add_argument(
        "--universe",
        default="nifty50",
        help="'nifty50' for the active universe, or a comma-separated symbol list (e.g. RELIANCE,TCS,INFY).",
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for plots and results files.")
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"], help="IC correlation method.")


def _run_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    params = _parse_params(args.params)

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        symbols = _resolve_universe_symbols(conn, args.universe)
        start_date, end_date = _resolve_date_range(conn, symbols, args.start, args.end)
        _ensure_forward_returns(conn, symbols)

        print(f"Screening '{args.signal}' | {len(symbols)} symbols | {start_date} to {end_date} | horizon={args.horizon}")
        result = screen_signal(
            conn, args.signal, params, args.horizon, start_date, end_date, symbols, args.method, output_dir
        )
        _print_single_result(result)
    finally:
        conn.close()
    return 0


def _batch_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    signal_names = [s.strip() for s in args.signals.split(",") if s.strip()]

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        symbols = _resolve_universe_symbols(conn, args.universe)
        start_date, end_date = _resolve_date_range(conn, symbols, args.start, args.end)
        _ensure_forward_returns(conn, symbols)
        history = _load_ohlcv_history(conn, symbols, end_date)

        print(
            f"Batch screening {len(signal_names)} signals | {len(symbols)} symbols "
            f"| {start_date} to {end_date} | horizon={args.horizon}"
        )
        results = []
        for signal_name in signal_names:
            print(f"  running {signal_name}...")
            result = screen_signal(
                conn, signal_name, {}, args.horizon, start_date, end_date, symbols, args.method, output_dir,
                history=history,
            )
            results.append(result)
            _print_single_result(result)

        table = _comparison_table(results)
        print(f"\n=== Batch comparison (horizon={args.horizon}, sorted by IC IR descending) ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"batch_{args.horizon}_{timestamp}.csv"
        table.to_csv(csv_path, index=False)
        print(f"\nSaved comparison table: {csv_path}")
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "batch":
        return _batch_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

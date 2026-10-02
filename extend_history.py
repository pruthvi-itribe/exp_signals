"""Extend stored history to 5 years and rebuild indicators + signals on top of it.

Re-fetches 5 years of daily OHLCV for the active universe (reusing
``bulk_fetch_and_store``), audits the extended range for trading-day gaps
(reusing ``audit_universe``), backfills ``indicators_daily`` for the new
range (``compute_and_store_all``), and regenerates signals for the full
range (``run_strategy``). Prints a summary of the resulting date coverage,
signal counts per strategy, and any symbols whose older history has a
materially worse gap rate than their recent history.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd

from src.indicators import compute_and_store_all
from src.strategy import run_strategy
from src.trading_calendar import audit_universe, get_nse_trading_days
from src.universe import DEFAULT_DB_PATH, bulk_fetch_and_store, get_active_universe
from strategies.registry import get_strategy

HISTORY_YEARS: int = 5
GAP_RATE_FLAG_THRESHOLD: float = 0.05


def _to_storage_symbols(yf_tickers: list[str]) -> list[str]:
    """Convert Yahoo Finance tickers (``RELIANCE.NS``) to storage symbols (``RELIANCE``)."""
    return [ticker.removesuffix(".NS") if ticker.endswith(".NS") else ticker for ticker in yf_tickers]


def _summarize_older_vs_recent_gaps(
    gaps_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Split ``audit_universe`` gaps at the window midpoint and flag worse-older-history symbols.

    Older history tends to have worse data quality (vendor backfills, early
    listing gaps, delistings/relistings), so this compares each symbol's gap
    rate in the first half of ``[start_date, end_date]`` against the second
    half. A symbol is flagged when its older-half gap rate exceeds its
    recent-half gap rate by more than ``GAP_RATE_FLAG_THRESHOLD`` (absolute).

    Args:
        gaps_df: Combined gaps DataFrame from ``audit_universe`` (columns
            ``symbol``, ``missing_date``, ``days_since_previous_available_date``).
        start_date: Inclusive lower bound of the audited window (``YYYY-MM-DD``).
        end_date: Inclusive upper bound of the audited window (``YYYY-MM-DD``).

    Returns:
        DataFrame with one row per symbol that has any gaps: ``symbol``,
        ``older_gap_count``, ``older_gap_rate``, ``recent_gap_count``,
        ``recent_gap_rate``, ``flagged`` — sorted by ``older_gap_rate`` descending.
    """
    empty_columns = [
        "symbol", "older_gap_count", "older_gap_rate",
        "recent_gap_count", "recent_gap_rate", "flagged",
    ]
    if gaps_df.empty:
        return pd.DataFrame(columns=empty_columns)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    midpoint = start_ts + (end_ts - start_ts) / 2
    midpoint_str = midpoint.strftime("%Y-%m-%d")

    expected_older = max(len(get_nse_trading_days(start_date, midpoint_str)), 1)
    expected_recent = max(len(get_nse_trading_days(midpoint_str, end_date)), 1)

    gaps = gaps_df.copy()
    gaps["missing_date"] = pd.to_datetime(gaps["missing_date"]).dt.normalize()
    gaps["period"] = gaps["missing_date"].apply(lambda d: "older" if d < midpoint else "recent")

    counts = (
        gaps.groupby(["symbol", "period"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["older", "recent"], fill_value=0)
    )

    summary = pd.DataFrame(
        {
            "symbol": counts.index,
            "older_gap_count": counts["older"].values,
            "older_gap_rate": counts["older"].values / expected_older,
            "recent_gap_count": counts["recent"].values,
            "recent_gap_rate": counts["recent"].values / expected_recent,
        }
    ).reset_index(drop=True)

    summary["flagged"] = (summary["older_gap_rate"] - summary["recent_gap_rate"]) > GAP_RATE_FLAG_THRESHOLD
    return summary.sort_values("older_gap_rate", ascending=False).reset_index(drop=True)


def main() -> None:
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        end = date.today()
        start = end - timedelta(days=365 * HISTORY_YEARS)
        start_date, end_date = start.isoformat(), end.isoformat()

        yf_tickers = get_active_universe(conn)
        if not yf_tickers:
            raise RuntimeError(
                "No active universe tickers found. Run `python -m src.universe` to sync Nifty 50 first."
            )
        storage_symbols = _to_storage_symbols(yf_tickers)

        print(f"=== Extending history to {HISTORY_YEARS} years for {len(yf_tickers)} active universe symbols ===")
        print(f"Requested range: {start_date} to {end_date}\n")

        fetch_summary = bulk_fetch_and_store(
            conn=conn,
            tickers=yf_tickers,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
        )
        print(
            f"\nFetch complete: {len(fetch_summary['successful'])} succeeded, "
            f"{len(fetch_summary['failed'])} failed."
        )

        print("\n--- Auditing for trading-day gaps across the full window ---")
        gaps_df = audit_universe(conn, storage_symbols, start_date, end_date)

        print("\n--- Backfilling indicators for the extended range ---")
        indicator_summary = compute_and_store_all(conn, symbols=storage_symbols)
        print(
            f"Indicators: {indicator_summary['symbols']} symbols, "
            f"{indicator_summary['ohlcv_rows']} OHLCV rows, "
            f"{indicator_summary['rows_inserted']} inserted / {indicator_summary['rows_updated']} updated."
        )

        print("\n--- Regenerating signals for the full range ---")
        run_strategy(conn, strategy=get_strategy("sma_crossover")(), symbols=storage_symbols)

        # --- Summary ---
        placeholders = ", ".join("?" for _ in storage_symbols)
        date_bounds = conn.execute(
            f"""
            SELECT MIN(timestamp)::DATE, MAX(timestamp)::DATE
            FROM ohlcv_data
            WHERE timeframe = '1d' AND symbol IN ({placeholders})
            """,
            storage_symbols,
        ).fetchone()

        signal_totals = conn.execute(
            """
            SELECT
                strategy,
                COUNT(*) AS total_signals,
                SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) AS buy_signals,
                SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) AS sell_signals
            FROM signals
            GROUP BY strategy
            ORDER BY strategy
            """
        ).df()

        gap_summary = _summarize_older_vs_recent_gaps(gaps_df, start_date, end_date)
        flagged = gap_summary.loc[gap_summary["flagged"]]

        print("\n=== Summary ===")
        print(f"Date range now available: {date_bounds[0]} to {date_bounds[1]}")

        print("\nTotal signal count per strategy:")
        if signal_totals.empty:
            print("  None")
        else:
            for row in signal_totals.itertuples(index=False):
                print(
                    f"  {row.strategy}: {int(row.total_signals)} "
                    f"(BUY={int(row.buy_signals)}, SELL={int(row.sell_signals)})"
                )

        print(
            f"\nSymbols with materially worse gap rates in older history "
            f"(older-half rate exceeds recent-half rate by more than {GAP_RATE_FLAG_THRESHOLD:.0%}):"
        )
        if flagged.empty:
            print("  None — older-history gap rates are not meaningfully worse than recent history.")
        else:
            for row in flagged.itertuples(index=False):
                print(
                    f"  {row.symbol}: older={row.older_gap_count} gaps ({row.older_gap_rate:.1%}), "
                    f"recent={row.recent_gap_count} gaps ({row.recent_gap_rate:.1%})"
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

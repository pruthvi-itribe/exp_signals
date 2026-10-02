"""Expand the tracked universe to Nifty 500 and extend history to 12 years.

Fetches ~12 years of daily OHLCV for the full active NIFTY500 universe using
the checkpointed, resumable fetcher (``src.universe.resumable_bulk_fetch``),
audits the result for trading-day gaps (reusing ``audit_universe`` and
``extend_history``'s older-vs-recent gap-rate flagging so that analysis isn't
duplicated), backfills ``indicators_daily`` for the expanded universe
(``compute_and_store_all``), and prints a detailed final summary including
full-vs-partial history coverage per symbol.

Safe to interrupt and re-run: ``resumable_bulk_fetch`` skips any symbol
already sufficiently covered in ``ohlcv_data`` rather than re-fetching it,
and logs every attempt to ``fetch_checkpoint`` — this script is designed to
be run in the background and resumed, not assumed to complete in one sitting.

KNOWN LIMITATION -- survivorship bias: the NIFTY500 universe this script
fetches history for reflects CURRENT index membership only (see
``fetch_nifty500_constituents`` in ``src/universe.py`` for the full
explanation). Projecting today's constituents back 12 years silently
excludes companies removed, delisted, renamed, or merged out of the index at
any point in that window. This is a deliberate, accepted tradeoff, not an
oversight -- printed here again as an explicit warning so it can't be
silently forgotten when interpreting backtest results built on this dataset.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd

from extend_history import GAP_RATE_FLAG_THRESHOLD, _summarize_older_vs_recent_gaps, _to_storage_symbols
from src.indicators import compute_and_store_all
from src.trading_calendar import audit_universe
from src.universe import DEFAULT_DB_PATH, SURVIVORSHIP_BIAS_WARNING, get_active_universe, resumable_bulk_fetch

HISTORY_YEARS: int = 12
COVERAGE_TOLERANCE_DAYS: int = 10
INDEX_NAME: str = "NIFTY500"


def main() -> None:
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        end = date.today()
        start = end - timedelta(days=365 * HISTORY_YEARS)
        start_date, end_date = start.isoformat(), end.isoformat()

        yf_tickers = get_active_universe(conn, index_name=INDEX_NAME)
        if not yf_tickers:
            raise RuntimeError(
                f"No active {INDEX_NAME} tickers found. Sync Nifty 500 constituents first "
                "(fetch_nifty500_constituents() + sync_universe(..., index_name='NIFTY500'))."
            )
        storage_symbols = _to_storage_symbols(yf_tickers)

        print(f"=== Extending history to {HISTORY_YEARS} years for {len(yf_tickers)} {INDEX_NAME} symbols ===")
        print(f"Requested range: {start_date} to {end_date}")
        print(f"\n{SURVIVORSHIP_BIAS_WARNING}\n")
        print(
            "Resumable/checkpointed fetch: safe to interrupt (Ctrl-C, timeout, crash) and "
            "re-run this script -- already-covered symbols are skipped, not re-fetched, and "
            "every attempt is logged to fetch_checkpoint.\n"
        )

        fetch_summary = resumable_bulk_fetch(
            conn=conn,
            tickers=yf_tickers,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            coverage_tolerance_days=COVERAGE_TOLERANCE_DAYS,
        )

        print("\n--- Auditing for trading-day gaps across the full window ---")
        gaps_df = audit_universe(conn, storage_symbols, start_date, end_date)

        gap_summary = _summarize_older_vs_recent_gaps(gaps_df, start_date, end_date)
        flagged = gap_summary.loc[gap_summary["flagged"]]
        print(
            f"\nSymbols with materially worse gap rates in OLDER history "
            f"(older-half rate exceeds recent-half rate by more than {GAP_RATE_FLAG_THRESHOLD:.0%} -- "
            "older data tends to have more quality issues; this flags where that pattern actually shows up "
            "rather than just reporting raw gap counts):"
        )
        if flagged.empty:
            print("  None -- older-history gap rates are not meaningfully worse than recent history.")
        else:
            for row in flagged.itertuples(index=False):
                print(
                    f"  {row.symbol}: older={row.older_gap_count} gaps ({row.older_gap_rate:.1%}), "
                    f"recent={row.recent_gap_count} gaps ({row.recent_gap_rate:.1%})"
                )

        print("\n--- Backfilling indicators for the expanded universe ---")
        indicator_summary = compute_and_store_all(conn, symbols=storage_symbols)
        print(
            f"Indicators: {indicator_summary['symbols']} symbols, "
            f"{indicator_summary['ohlcv_rows']} OHLCV rows, "
            f"{indicator_summary['rows_inserted']} inserted / {indicator_summary['rows_updated']} updated."
        )

        # --- Final counts ---
        universe_symbol_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM universe").fetchone()[0]

        placeholders = ", ".join("?" for _ in storage_symbols)
        ohlcv_scoped = conn.execute(
            f"SELECT COUNT(*) FROM ohlcv_data WHERE timeframe = '1d' AND symbol IN ({placeholders})",
            storage_symbols,
        ).fetchone()[0]
        ohlcv_grand_total = conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0]

        indicators_scoped = conn.execute(
            f"SELECT COUNT(*) FROM indicators_daily WHERE symbol IN ({placeholders})",
            storage_symbols,
        ).fetchone()[0]
        indicators_grand_total = conn.execute("SELECT COUNT(*) FROM indicators_daily").fetchone()[0]

        coverage_df = conn.execute(
            f"""
            SELECT symbol, MIN(timestamp)::DATE AS min_date, MAX(timestamp)::DATE AS max_date
            FROM ohlcv_data
            WHERE timeframe = '1d' AND symbol IN ({placeholders})
            GROUP BY symbol
            """,
            storage_symbols,
        ).df()
        coverage_df["min_date"] = pd.to_datetime(coverage_df["min_date"])
        full_cutoff = pd.Timestamp(start_date) + pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS)
        full_coverage = coverage_df.loc[coverage_df["min_date"] <= full_cutoff]
        partial_coverage = coverage_df.loc[coverage_df["min_date"] > full_cutoff]
        missing_symbols = sorted(set(storage_symbols) - set(coverage_df["symbol"]))

        print("\n=== Final Summary ===")
        print(f"Total distinct symbols in universe table (all indices): {universe_symbol_count}")
        print(f"Active {INDEX_NAME} symbols: {len(storage_symbols)}")
        print(f"ohlcv_data rows -- {INDEX_NAME} symbols: {ohlcv_scoped}  |  whole table: {ohlcv_grand_total}")
        print(f"indicators_daily rows -- {INDEX_NAME} symbols: {indicators_scoped}  |  whole table: {indicators_grand_total}")
        print(f"\n{INDEX_NAME} symbols with any data: {len(coverage_df)} / {len(storage_symbols)}")
        print(f"  Full {HISTORY_YEARS}-year coverage:  {len(full_coverage)}")
        print(f"  Partial coverage (shorter listing history): {len(partial_coverage)}")
        if not partial_coverage.empty:
            print("  Partial-coverage symbols (earliest available date):")
            for row in partial_coverage.sort_values("min_date", ascending=False).itertuples(index=False):
                print(f"    {row.symbol}: {row.min_date.date()}")
        if missing_symbols:
            print(f"  Symbols with NO data at all (fetch failed -- see fetch_checkpoint): {len(missing_symbols)}")
            print(f"    {', '.join(missing_symbols)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

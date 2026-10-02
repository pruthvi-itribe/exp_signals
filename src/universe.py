"""
Universe management for tracked Indian equity indices.

Maintains a historical log of index constituents (starting with Nifty 50),
syncs live constituent lists against DuckDB, bulk-fetches OHLCV data, and
audits coverage gaps before downstream indicator work.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

import duckdb
import pandas as pd
import requests
import yfinance as yf

from src.ingestion.historical_fetcher import HistoricalFetcher
from src.storage.upsert import upsert_ohlcv
from src.validation.validate_ohlcv import log_issues, validate_ohlcv
from src.trading_calendar import audit_universe

# Live sources (same official CSV mirrored in two places). NSE archives has proven
# more reliable than niftyindices.com in practice (fewer timeouts); niftyindices
# is kept as secondary. Last resort: data/nifty50_manual.csv.
NIFTY50_CSV_URLS: tuple[str, ...] = (
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
)
MANUAL_NIFTY50_CSV: Path = Path("data/nifty50_manual.csv")

# Same NSE archive/niftyindices mirror pattern as Nifty 50, just the broader index.
# See fetch_nifty500_constituents()'s docstring for the survivorship-bias caveat
# that applies when this universe is used to backfill many years of history.
NIFTY500_CSV_URLS: tuple[str, ...] = (
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)
MANUAL_NIFTY500_CSV: Path = Path("data/nifty500_manual.csv")

DEFAULT_DB_PATH: Path = Path("data/trading_data.duckdb")
FETCH_DELAY_SECONDS: float = 0.75
# resumable_bulk_fetch() targets much larger batches (e.g. 500 symbols x 12 years)
# run over a much longer wall-clock time than bulk_fetch_and_store()'s typical use,
# so it defaults to a gentler pace to reduce the odds of yfinance rate-limiting
# partway through a long run.
RESUMABLE_FETCH_DELAY_SECONDS: float = 2.0

SURVIVORSHIP_BIAS_WARNING: str = (
    "WARNING: Nifty 500 constituents reflect CURRENT index membership only. "
    "Companies that were removed, delisted, renamed, or merged out of the index "
    "at any point during a historical lookback window are NOT included "
    "(survivorship bias). This is a deliberate, accepted tradeoff for now -- "
    "point-in-time historical index membership isn't freely available. Keep this "
    "in mind when interpreting any backtest results built on this dataset; results "
    "will look somewhat better than a true point-in-time universe would have produced."
)

ConstituentRecord = TypedDict("ConstituentRecord", {"symbol": str, "company_name": str})


def ensure_universe_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``universe`` table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS universe (
            symbol       VARCHAR NOT NULL,
            yf_ticker    VARCHAR NOT NULL,
            index_name   VARCHAR NOT NULL,
            added_date   DATE NOT NULL,
            removed_date DATE,
            is_active    BOOLEAN DEFAULT true,
            PRIMARY KEY (symbol, index_name, added_date)
        )
        """
    )


def _parse_constituent_csv(csv_text: str) -> list[ConstituentRecord]:
    """Parse an NSE index constituent CSV (Nifty 50, Nifty 500, ...) into normalized records."""
    frame = pd.read_csv(io.StringIO(csv_text))
    symbol_column = "Symbol" if "Symbol" in frame.columns else "symbol"
    name_column = "Company Name" if "Company Name" in frame.columns else "company_name"

    if symbol_column not in frame.columns:
        raise ValueError("CSV is missing a symbol column.")

    records: list[ConstituentRecord] = []
    for _, row in frame.iterrows():
        symbol = str(row[symbol_column]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        company_name = str(row.get(name_column, "")).strip()
        records.append({"symbol": symbol, "company_name": company_name})

    if not records:
        raise ValueError("No constituents parsed from CSV.")

    return records


def _load_manual_constituent_csv(path: Path) -> list[ConstituentRecord]:
    """Load constituents from a manually maintained fallback CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Manual fallback CSV not found at {path}. "
            "Create it with columns: symbol, company_name."
        )

    frame = pd.read_csv(path)
    symbol_column = "symbol" if "symbol" in frame.columns else "Symbol"
    name_column = "company_name" if "company_name" in frame.columns else "Company Name"

    records: list[ConstituentRecord] = []
    for _, row in frame.iterrows():
        symbol = str(row[symbol_column]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        company_name = str(row.get(name_column, "")).strip()
        records.append({"symbol": symbol, "company_name": company_name})

    if not records:
        raise ValueError(f"No constituents found in manual CSV: {path}")

    return records


def _fetch_index_constituents(
    csv_urls: tuple[str, ...],
    manual_csv_path: Path,
    index_label: str,
) -> list[dict[str, str]]:
    """Shared fetch-with-fallback logic for any NSE index constituent CSV.

    Tries each URL in ``csv_urls`` in order, falling back to
    ``manual_csv_path`` if every live source fails. Used by both
    ``fetch_nifty50_constituents`` and ``fetch_nifty500_constituents`` — the
    only difference between the two indices is which URLs/fallback path/
    label to use, so that's all each of those two thin wrappers supplies.

    Returns:
        List of dicts with keys ``symbol``, ``company_name``, and ``yf_ticker``.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; trading-bot/1.0; +https://github.com/local/trading-bot)"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }
    last_error: Exception | None = None

    for url in csv_urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            records = _parse_constituent_csv(response.text)
            print(f"Fetched {index_label} constituents from {url} ({len(records)} symbols).")
            return [
                {
                    "symbol": record["symbol"],
                    "company_name": record["company_name"],
                    "yf_ticker": f"{record['symbol']}.NS",
                }
                for record in records
            ]
        except Exception as exc:
            last_error = exc
            print(f"Failed to fetch {index_label} from {url}: {exc}")

    try:
        records = _load_manual_constituent_csv(manual_csv_path)
        print(
            f"Live fetch failed; loaded {len(records)} {index_label} symbols "
            f"from manual fallback {manual_csv_path}."
        )
        if last_error is not None:
            print(f"Last live-fetch error: {last_error}")
        return [
            {
                "symbol": record["symbol"],
                "company_name": record["company_name"],
                "yf_ticker": f"{record['symbol']}.NS",
            }
            for record in records
        ]
    except Exception as manual_exc:
        raise RuntimeError(
            f"Unable to fetch {index_label} constituents from live sources or manual fallback."
        ) from manual_exc


def fetch_nifty50_constituents() -> list[dict[str, str]]:
    """Fetch the current Nifty 50 constituent list.

    Attempts live CSV downloads from NSE Indices (primary) and NSE archives
    (secondary). If both fail, falls back to ``data/nifty50_manual.csv``.

    Returns:
        List of dicts with keys ``symbol``, ``company_name``, and ``yf_ticker``.
    """
    return _fetch_index_constituents(NIFTY50_CSV_URLS, MANUAL_NIFTY50_CSV, "Nifty 50")


def fetch_nifty500_constituents() -> list[dict[str, str]]:
    """Fetch the current Nifty 500 constituent list.

    Attempts live CSV downloads from NSE archives (primary) and NSE Indices
    (secondary). If both fail, falls back to ``data/nifty500_manual.csv``.

    KNOWN LIMITATION — survivorship bias: this returns CURRENT Nifty 500
    membership only. Point-in-time historical index membership (which
    stocks were actually in the index at any given past date) isn't freely
    available, so backfilling many years of history against this universe
    silently excludes companies that were removed, delisted, renamed, or
    merged out of the index at some point in that window. This is a
    deliberate, accepted tradeoff, not an oversight — printed as an explicit
    warning every time this function runs (see ``SURVIVORSHIP_BIAS_WARNING``)
    so it can't be silently forgotten later when interpreting backtest
    results built on this dataset.

    Returns:
        List of dicts with keys ``symbol``, ``company_name``, and ``yf_ticker``.
    """
    print(SURVIVORSHIP_BIAS_WARNING)
    return _fetch_index_constituents(NIFTY500_CSV_URLS, MANUAL_NIFTY500_CSV, "Nifty 500")


def sync_universe(
    conn: duckdb.DuckDBPyConnection,
    constituents: list[dict[str, str]],
    index_name: str = "NIFTY50",
) -> dict[str, int]:
    """Sync index constituents into the historical ``universe`` table.

    Inserts newly added symbols, marks removed symbols inactive, and never
    deletes historical rows.

    Args:
        conn: Open DuckDB connection.
        constituents: Current constituent list (must include ``symbol`` and
            ``yf_ticker`` keys).
        index_name: Index label stored in ``universe.index_name``.

    Returns:
        Summary with counts of added and removed symbols.
    """
    ensure_universe_schema(conn)
    today = date.today()
    new_symbols = {item["symbol"] for item in constituents}
    constituent_by_symbol = {item["symbol"]: item for item in constituents}

    active_rows = conn.execute(
        """
        SELECT symbol
        FROM universe
        WHERE index_name = ? AND is_active = true
        """,
        [index_name],
    ).fetchall()
    active_symbols = {row[0] for row in active_rows}

    removed_symbols = sorted(active_symbols - new_symbols)
    for symbol in removed_symbols:
        conn.execute(
            """
            UPDATE universe
            SET removed_date = ?, is_active = false
            WHERE symbol = ? AND index_name = ? AND is_active = true
            """,
            [today, symbol, index_name],
        )

    added_symbols = sorted(new_symbols - active_symbols)
    for symbol in added_symbols:
        item = constituent_by_symbol[symbol]
        conn.execute(
            """
            INSERT INTO universe (
                symbol, yf_ticker, index_name, added_date, removed_date, is_active
            )
            VALUES (?, ?, ?, ?, NULL, true)
            """,
            [symbol, item["yf_ticker"], index_name, today],
        )

    return {
        "added": len(added_symbols),
        "removed": len(removed_symbols),
        "active": len(new_symbols),
    }


def get_active_universe(
    conn: duckdb.DuckDBPyConnection,
    index_name: str = "NIFTY50",
    as_of_date: str | None = None,
) -> list[str]:
    """Return Yahoo Finance tickers active in the universe on a given date.

    Args:
        conn: Open DuckDB connection.
        index_name: Index label to filter on.
        as_of_date: ISO date string (``YYYY-MM-DD``). Defaults to today.

    Returns:
        Sorted list of ``yf_ticker`` values (for example, ``RELIANCE.NS``).
    """
    ensure_universe_schema(conn)
    reference_date = as_of_date or date.today().isoformat()

    rows = conn.execute(
        """
        SELECT yf_ticker
        FROM universe
        WHERE index_name = ?
          AND added_date <= ?
          AND (removed_date IS NULL OR removed_date > ?)
        ORDER BY yf_ticker
        """,
        [index_name, reference_date, reference_date],
    ).fetchall()

    return [row[0] for row in rows]


def _filter_error_severity_rows(
    candles: pd.DataFrame,
    issues_df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop candle rows flagged with ``severity='error'`` before upsert."""
    if issues_df.empty or candles.empty:
        return candles

    error_issues = issues_df.loc[issues_df["severity"] == "error"]
    if error_issues.empty:
        return candles

    error_dates = pd.to_datetime(error_issues["date"], errors="coerce").dt.normalize()
    candle_dates = pd.to_datetime(candles["timestamp"], errors="coerce").dt.normalize()
    keep_mask = ~candle_dates.isin(error_dates.unique())
    return candles.loc[keep_mask].copy()


def _fetch_yfinance_history(
    yf_ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pd.DataFrame:
    """Download raw OHLCV history for one ticker."""
    return yf.Ticker(yf_ticker).history(
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
    )


class FetchOutcome(TypedDict):
    """Result of fetching, validating, and upserting one ticker's history."""

    success: bool
    rows: int
    first_date: str | None
    last_date: str | None
    error: str | None


def _fetch_validate_upsert_one(
    conn: duckdb.DuckDBPyConnection,
    yf_ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> FetchOutcome:
    """Fetch, validate, and upsert OHLCV history for one ticker.

    This is the shared core used by both ``bulk_fetch_and_store`` (simple
    loop, no checkpointing — fine for small batches) and
    ``resumable_bulk_fetch`` (checkpointed, skip-aware — for batches large
    enough to span multiple runs), so the actual fetch/validate/upsert logic
    lives in exactly one place rather than being duplicated between them.

    Args:
        conn: Open DuckDB connection.
        yf_ticker: Yahoo Finance ticker (for example, ``RELIANCE.NS``).
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Exclusive end date passed to yfinance (``YYYY-MM-DD``).
        interval: Candle interval (for example, ``1d``).

    Returns:
        A ``FetchOutcome``. On success, ``first_date``/``last_date`` are the
        actual range of the rows just fetched and stored (not necessarily
        matching the requested range — a symbol with a shorter listing
        history than requested is expected and still reported as success
        with its actual, shorter range).
    """
    storage_symbol = HistoricalFetcher._to_storage_symbol(yf_ticker)
    try:
        raw_df = _fetch_yfinance_history(
            yf_ticker=yf_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if raw_df.empty:
            raise ValueError("yfinance returned no rows.")

        candles = HistoricalFetcher._prepare_candles(raw_df, storage_symbol, interval)
        issues_df = validate_ohlcv(candles, storage_symbol)
        if not issues_df.empty:
            log_issues(conn, issues_df)

        clean_candles = _filter_error_severity_rows(candles, issues_df)
        if clean_candles.empty:
            raise ValueError("All rows failed validation with error severity.")

        upsert_ohlcv(conn, clean_candles, timeframe=interval)

        timestamps = pd.to_datetime(clean_candles["timestamp"])
        return {
            "success": True,
            "rows": len(clean_candles),
            "first_date": timestamps.min().date().isoformat(),
            "last_date": timestamps.max().date().isoformat(),
            "error": None,
        }
    except Exception as exc:
        return {"success": False, "rows": 0, "first_date": None, "last_date": None, "error": str(exc)}


def bulk_fetch_and_store(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    delay_seconds: float = FETCH_DELAY_SECONDS,
) -> dict[str, list[str]]:
    """Fetch, validate, and upsert OHLCV data for many tickers.

    Args:
        conn: Open DuckDB connection.
        tickers: Yahoo Finance tickers (for example, ``RELIANCE.NS``).
        start_date: Inclusive start date (``YYYY-MM-DD``).
        end_date: Exclusive end date passed to yfinance (``YYYY-MM-DD``).
        interval: Candle interval (default ``1d``).
        delay_seconds: Pause between ticker requests to reduce rate limiting.

    Returns:
        Dictionary with ``successful`` and ``failed`` ticker lists.
    """
    try:
        from tqdm import tqdm

        iterator: object = tqdm(tickers, desc="Fetching tickers", unit="ticker")
    except ImportError:
        iterator = tickers

    successful: list[str] = []
    failed: list[str] = []

    for index, yf_ticker in enumerate(iterator, start=1):
        storage_symbol = HistoricalFetcher._to_storage_symbol(yf_ticker)
        if not hasattr(iterator, "set_description"):
            print(f"[{index}/{len(tickers)}] Fetching {storage_symbol} ({yf_ticker})...")

        outcome = _fetch_validate_upsert_one(conn, yf_ticker, start_date, end_date, interval)
        if outcome["success"]:
            successful.append(yf_ticker)
        else:
            failed.append(yf_ticker)
            print(f"Failed {yf_ticker}: {outcome['error']}")

        if index < len(tickers):
            time.sleep(delay_seconds)

    if failed:
        print("\nTickers that failed and may need manual retry:")
        for ticker in failed:
            print(f"  - {ticker}")

    return {"successful": successful, "failed": failed}


def ensure_fetch_checkpoint_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``fetch_checkpoint`` table if it does not already exist.

    One row per symbol, upserted on every attempt — this is a *current
    status* journal for a resumable fetch job, not a full attempt history.
    ``status`` is one of ``'pending'`` (write happens right before a fetch
    attempt starts, so a row stuck on ``'pending'`` after an interrupted run
    tells you exactly which symbol was in flight when things stopped),
    ``'success'``, ``'failed'``, or ``'skipped'`` (already had sufficient
    coverage in ``ohlcv_data`` before this run even attempted it).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_checkpoint (
            symbol        VARCHAR NOT NULL,
            yf_ticker     VARCHAR NOT NULL,
            status        VARCHAR NOT NULL,
            rows_fetched  BIGINT,
            first_date    DATE,
            last_date     DATE,
            error_message VARCHAR,
            attempted_at  TIMESTAMP NOT NULL,
            PRIMARY KEY (symbol)
        )
        """
    )


def _write_checkpoint(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    yf_ticker: str,
    status: str,
    rows: int | None,
    first_date: str | None,
    last_date: str | None,
    error: str | None,
) -> None:
    """Upsert one row into ``fetch_checkpoint``, overwriting any prior attempt for this symbol."""
    conn.execute(
        """
        INSERT INTO fetch_checkpoint (
            symbol, yf_ticker, status, rows_fetched, first_date, last_date, error_message, attempted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)
        ON CONFLICT (symbol) DO UPDATE SET
            yf_ticker = excluded.yf_ticker,
            status = excluded.status,
            rows_fetched = excluded.rows_fetched,
            first_date = excluded.first_date,
            last_date = excluded.last_date,
            error_message = excluded.error_message,
            attempted_at = excluded.attempted_at
        """,
        [symbol, yf_ticker, status, rows, first_date, last_date, error],
    )


def _has_sufficient_coverage(
    conn: duckdb.DuckDBPyConnection,
    storage_symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    tolerance_days: int,
) -> bool:
    """Check whether ``ohlcv_data`` already covers the requested range for this symbol.

    "Covers" means: the earliest stored date is on/before ``start_date``
    (within ``tolerance_days``, so a symbol's actual listing date landing a
    little after the requested start isn't treated as missing data — the
    symbol just didn't exist yet) AND the latest stored date is on/after
    ``end_date`` minus the same tolerance (allowing for the most recent few
    trading days not having been fetched yet). A symbol with data that
    doesn't reach back far enough is NOT considered sufficiently covered —
    it's re-fetched for its full requested range, since partial coverage
    here doesn't tell us whether the gap is a genuine listing-date limit or
    just data nobody has fetched yet.

    Args:
        conn: Open DuckDB connection.
        storage_symbol: Symbol as stored in ``ohlcv_data`` (no ``.NS`` suffix).
        start_date: Inclusive requested start date (``YYYY-MM-DD``).
        end_date: Requested end date (``YYYY-MM-DD``).
        interval: Candle interval (for example, ``1d``).
        tolerance_days: Slack, in days, applied to both ends of the range.

    Returns:
        ``True`` if existing data is judged sufficient (fetch can be skipped).
    """
    row = conn.execute(
        "SELECT MIN(timestamp)::DATE, MAX(timestamp)::DATE FROM ohlcv_data WHERE symbol = ? AND timeframe = ?",
        [storage_symbol, interval],
    ).fetchone()
    existing_min, existing_max = row
    if existing_min is None:
        return False

    tolerance = pd.Timedelta(days=tolerance_days)
    covers_start = pd.Timestamp(existing_min) <= pd.Timestamp(start_date) + tolerance
    covers_end = pd.Timestamp(existing_max) >= pd.Timestamp(end_date) - tolerance
    return bool(covers_start and covers_end)


def resumable_bulk_fetch(
    conn: duckdb.DuckDBPyConnection,
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    delay_seconds: float = RESUMABLE_FETCH_DELAY_SECONDS,
    coverage_tolerance_days: int = 10,
) -> dict[str, object]:
    """Resumable, checkpointed variant of ``bulk_fetch_and_store`` for large batches.

    Built for jobs too big to reliably complete in one sitting (e.g. 500
    symbols x 12 years of daily history) — safe to interrupt (Ctrl-C,
    timeout, crash) and re-run: any symbol already sufficiently covered in
    ``ohlcv_data`` (see ``_has_sufficient_coverage``) is skipped rather than
    re-fetched, and every attempt (success, failure, or skip) is logged to
    ``fetch_checkpoint`` so progress across multiple runs stays visible. The
    skip decision is always made against ``ohlcv_data`` itself (the source
    of truth for what's actually been fetched), not against
    ``fetch_checkpoint``, which is a status log, not the mechanism deciding
    what to skip.

    Uses the exact same per-ticker fetch/validate/upsert path as
    ``bulk_fetch_and_store`` (``_fetch_validate_upsert_one``) — this
    function only adds the skip-check and checkpoint logging around it; it
    does not duplicate the fetch logic.

    Partial history (for example a symbol that IPO'd 3 years ago, so 12
    years of daily bars isn't available) is expected and NOT treated as a
    failure — whatever yfinance actually returns is fetched and stored, and
    the checkpoint records the actual range obtained so partial coverage
    stays visible rather than looking identical to full coverage.

    Args:
        conn: Open DuckDB connection.
        tickers: Yahoo Finance tickers to fetch.
        start_date: Inclusive requested start date (``YYYY-MM-DD``).
        end_date: Exclusive requested end date passed to yfinance (``YYYY-MM-DD``).
        interval: Candle interval (default ``1d``).
        delay_seconds: Pause between requests. Defaults higher than
            ``bulk_fetch_and_store``'s ``FETCH_DELAY_SECONDS`` (see
            ``RESUMABLE_FETCH_DELAY_SECONDS``) since a run of hundreds of
            symbols over a long wall-clock time is far more likely to trip
            yfinance rate limiting than a short, small-batch run.
        coverage_tolerance_days: Tolerance (days) used both when deciding
            whether existing data already covers the requested range (see
            ``_has_sufficient_coverage``) and when classifying a successful
            fetch as "full" vs. "partial" range in the returned summary.

    Returns:
        Summary dict with ``succeeded_full`` (yf_tickers fetched with data
        reaching back to the requested start, within tolerance),
        ``succeeded_partial`` (fetched successfully but with less history
        than requested), ``skipped`` (already had sufficient coverage),
        ``failed`` (list of ``(yf_ticker, error)`` tuples), and
        ``rows_added`` (total rows fetched across all successful tickers in
        *this* run — not counting rows that already existed and were skipped).
    """
    ensure_fetch_checkpoint_schema(conn)

    try:
        from tqdm import tqdm

        iterator: object = tqdm(tickers, desc="Resumable fetch", unit="ticker")
    except ImportError:
        iterator = tickers

    succeeded_full: list[str] = []
    succeeded_partial: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    rows_added = 0

    start_ts = pd.Timestamp(start_date)
    full_range_cutoff = start_ts + pd.Timedelta(days=coverage_tolerance_days)

    for index, yf_ticker in enumerate(iterator, start=1):
        storage_symbol = HistoricalFetcher._to_storage_symbol(yf_ticker)
        if not hasattr(iterator, "set_description"):
            print(f"[{index}/{len(tickers)}] {storage_symbol} ({yf_ticker})...")

        if _has_sufficient_coverage(conn, storage_symbol, start_date, end_date, interval, coverage_tolerance_days):
            skipped.append(yf_ticker)
            _write_checkpoint(conn, storage_symbol, yf_ticker, "skipped", None, None, None, None)
            continue

        _write_checkpoint(conn, storage_symbol, yf_ticker, "pending", None, None, None, None)
        outcome = _fetch_validate_upsert_one(conn, yf_ticker, start_date, end_date, interval)

        if outcome["success"]:
            rows_added += outcome["rows"]
            if pd.Timestamp(outcome["first_date"]) <= full_range_cutoff:
                succeeded_full.append(yf_ticker)
            else:
                succeeded_partial.append(yf_ticker)
            _write_checkpoint(
                conn, storage_symbol, yf_ticker, "success",
                outcome["rows"], outcome["first_date"], outcome["last_date"], None,
            )
        else:
            failed.append((yf_ticker, outcome["error"]))
            _write_checkpoint(conn, storage_symbol, yf_ticker, "failed", None, None, None, outcome["error"])
            print(f"Failed {yf_ticker}: {outcome['error']}")

        if index < len(tickers):
            time.sleep(delay_seconds)

    print("\n=== resumable_bulk_fetch summary ===")
    print(f"Total processed:     {len(tickers)}")
    print(f"Succeeded (full):    {len(succeeded_full)}")
    print(f"Succeeded (partial): {len(succeeded_partial)}")
    print(f"Skipped (had data):  {len(skipped)}")
    print(f"Failed:              {len(failed)}")
    print(f"Rows added this run: {rows_added}")
    if succeeded_partial:
        print("\nPartial-range tickers (likely recent listings — check fetch_checkpoint for actual dates):")
        for ticker in succeeded_partial:
            print(f"  - {ticker}")
    if failed:
        print("\nFailed tickers (see fetch_checkpoint.error_message for detail):")
        for ticker, error in failed:
            print(f"  - {ticker}: {error}")

    return {
        "succeeded_full": succeeded_full,
        "succeeded_partial": succeeded_partial,
        "skipped": skipped,
        "failed": failed,
        "rows_added": rows_added,
    }


def _to_storage_symbols(yf_tickers: list[str]) -> list[str]:
    """Convert Yahoo Finance tickers to symbols stored in ``ohlcv_data``."""
    return [HistoricalFetcher._to_storage_symbol(ticker) for ticker in yf_tickers]


if __name__ == "__main__":
    end = date.today()
    start = end - timedelta(days=365)
    start_date_str = start.isoformat()
    end_date_str = end.isoformat()

    print("=== Nifty 50 Universe Sync & Bulk Fetch ===")
    constituents = fetch_nifty50_constituents()
    print(f"Constituents fetched: {len(constituents)}")

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        sync_summary = sync_universe(conn, constituents, index_name="NIFTY50")
        print(
            f"Universe sync complete: added={sync_summary['added']}, "
            f"removed={sync_summary['removed']}, active={sync_summary['active']}"
        )

        active_tickers = get_active_universe(conn, index_name="NIFTY50")
        print(f"Active universe size: {len(active_tickers)}")

        fetch_summary = bulk_fetch_and_store(
            conn=conn,
            tickers=active_tickers,
            start_date=start_date_str,
            end_date=end_date_str,
            interval="1d",
        )

        storage_symbols = _to_storage_symbols(active_tickers)
        gaps_df = audit_universe(
            conn,
            storage_symbols,
            start_date_str,
            end_date_str,
        )

        gap_symbols = (
            sorted(gaps_df["symbol"].unique().tolist()) if not gaps_df.empty else []
        )

        print("\n=== Final Summary ===")
        print(f"Successfully fetched: {len(fetch_summary['successful'])} tickers")
        print(f"Failed fetch: {len(fetch_summary['failed'])} tickers")
        if fetch_summary["failed"]:
            print("Failed tickers:", ", ".join(fetch_summary["failed"]))
        print(f"Tickers with data gaps: {len(gap_symbols)}")
        if gap_symbols:
            print("Gap tickers:", ", ".join(gap_symbols))
        else:
            print("No trading-day gaps detected across the active universe.")
    finally:
        conn.close()

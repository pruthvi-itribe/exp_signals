"""Upsert helpers for OHLCV candle data in DuckDB."""

from __future__ import annotations

import duckdb
import pandas as pd

UPSERT_INPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

UpsertSummary = dict[str, int]


def ensure_ohlcv_upsert_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure ``ohlcv_data`` exists and includes ``fetched_at`` for upsert tracking."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            timeframe VARCHAR NOT NULL,
            fetched_at TIMESTAMP,
            PRIMARY KEY (symbol, timestamp, timeframe)
        )
        """
    )
    conn.execute("ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS adj_close DOUBLE")
    conn.execute("ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMP")


def _prepare_staging_dataframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Attach ``timeframe`` and ``fetched_at`` to the validated input batch."""
    missing_columns = [col for col in UPSERT_INPUT_COLUMNS if col not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    staging = df.loc[:, UPSERT_INPUT_COLUMNS].copy()
    staging["timestamp"] = pd.to_datetime(staging["timestamp"])
    staging["timeframe"] = timeframe
    staging["fetched_at"] = pd.Timestamp.now("UTC").tz_localize(None)
    return staging


def upsert_ohlcv(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    timeframe: str = "1d",
) -> UpsertSummary:
    """Insert or update OHLCV rows, refreshing price fields and ``fetched_at`` on conflict.

    Conflicts are resolved on the primary key ``(symbol, timestamp, timeframe)``.
    The input batch is expected to have already passed validation; warnings do not
    block insertion.

    Args:
        conn: Open DuckDB connection.
        df: Validated OHLCV DataFrame with ``symbol``, ``timestamp``, OHLC,
            ``adj_close``, and ``volume`` columns.
        timeframe: Candle interval label applied to every row in the batch.

    Returns:
        Summary dictionary with ``rows_inserted``, ``rows_updated``, and
        ``total_rows`` counts.
    """
    empty_summary: UpsertSummary = {
        "rows_inserted": 0,
        "rows_updated": 0,
        "total_rows": 0,
    }
    if df.empty:
        return empty_summary

    ensure_ohlcv_upsert_schema(conn)
    staging = _prepare_staging_dataframe(df, timeframe)
    total_rows = len(staging)

    conn.register("_upsert_staging", staging)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            rows_updated = conn.execute(
                """
                SELECT COUNT(*)::BIGINT
                FROM _upsert_staging AS staging
                INNER JOIN ohlcv_data AS existing
                    ON staging.symbol = existing.symbol
                   AND staging.timestamp = existing.timestamp
                   AND staging.timeframe = existing.timeframe
                """
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO ohlcv_data (
                    symbol,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume,
                    timeframe,
                    fetched_at
                )
                SELECT
                    symbol,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    adj_close,
                    volume,
                    timeframe,
                    fetched_at
                FROM _upsert_staging
                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    adj_close = excluded.adj_close,
                    volume = excluded.volume,
                    fetched_at = excluded.fetched_at
                """
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.unregister("_upsert_staging")

    rows_inserted = total_rows - int(rows_updated)
    return {
        "rows_inserted": rows_inserted,
        "rows_updated": int(rows_updated),
        "total_rows": total_rows,
    }


def _build_demo_dataframe(
    rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Build a small OHLCV DataFrame for the module demo."""
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with duckdb.connect(":memory:") as conn:
        ensure_ohlcv_upsert_schema(conn)

        initial_df = _build_demo_dataframe(
            [
                {
                    "symbol": "RELIANCE",
                    "timestamp": pd.Timestamp("2024-01-01"),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 104.0,
                    "adj_close": 104.0,
                    "volume": 1000,
                },
                {
                    "symbol": "RELIANCE",
                    "timestamp": pd.Timestamp("2024-01-02"),
                    "open": 101.0,
                    "high": 106.0,
                    "low": 100.0,
                    "close": 105.0,
                    "adj_close": 105.0,
                    "volume": 1200,
                },
            ]
        )

        first_summary = upsert_ohlcv(conn, initial_df, timeframe="1d")
        print("First upsert summary:", first_summary)

        refresh_df = _build_demo_dataframe(
            [
                {
                    "symbol": "RELIANCE",
                    "timestamp": pd.Timestamp("2024-01-02"),
                    "open": 101.5,
                    "high": 107.0,
                    "low": 100.5,
                    "close": 106.5,
                    "adj_close": 106.5,
                    "volume": 1500,
                },
                {
                    "symbol": "RELIANCE",
                    "timestamp": pd.Timestamp("2024-01-03"),
                    "open": 102.0,
                    "high": 108.0,
                    "low": 101.0,
                    "close": 107.0,
                    "adj_close": 107.0,
                    "volume": 1300,
                },
            ]
        )

        second_summary = upsert_ohlcv(conn, refresh_df, timeframe="1d")
        print("Second upsert summary:", second_summary)

        final_table = conn.execute(
            """
            SELECT symbol, timestamp, open, high, low, close, adj_close, volume, timeframe, fetched_at
            FROM ohlcv_data
            ORDER BY timestamp
            """
        ).df()
        print("\nFinal ohlcv_data contents:")
        print(final_table.to_string(index=False))

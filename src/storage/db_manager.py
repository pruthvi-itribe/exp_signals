"""DuckDB-backed time-series storage for OHLCV candle data."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
from src.storage.upsert import upsert_ohlcv

OHLCV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "timeframe",
)


class DatabaseManager:
    """Manage local OHLCV time-series storage in DuckDB."""

    DEFAULT_DB_PATH: Path = Path("data/trading_data.duckdb")

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the database manager and ensure schema exists.

        Args:
            db_path: Path to the DuckDB file. Defaults to ``data/trading_data.duckdb``.
        """
        self.db_path = Path(db_path) if db_path is not None else self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield a DuckDB connection that is closed when the context exits."""
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create the ``ohlcv_data`` table if it does not already exist."""
        with self.get_connection() as conn:
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

    def save_candles(self, df: pd.DataFrame, timeframe: str = "1d") -> None:
        """Insert or update OHLCV candles using advanced upsert logic.

        Args:
            df: DataFrame containing OHLCV columns.
            timeframe: Candle interval label.

        Raises:
            ValueError: If required columns are missing from ``df``.
        """
        if df.empty:
            return

        with self.get_connection() as conn:
            upsert_ohlcv(conn, df, timeframe=timeframe)

    def fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch stored candles for a symbol and timeframe.

        Args:
            symbol: Ticker symbol (for example, ``RELIANCE.NS``).
            timeframe: Candle interval label (for example, ``1d`` or ``1h``).

        Returns:
            DataFrame ordered by ``timestamp`` ascending. Empty if no rows match.
        """
        with self.get_connection() as conn:
            result = conn.execute(
                #"""
                #SELECT symbol, timestamp, open, high, low, close, adj_close, volume, timeframe
                """
                SELECT *
                FROM ohlcv_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp ASC
                """,
                [symbol, timeframe],
            )
            return result.df()

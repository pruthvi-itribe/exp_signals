"""Download and persist historical OHLCV data for Indian equities via yfinance."""

from __future__ import annotations

import yfinance as yf
import pandas as pd

from src.storage.db_manager import DatabaseManager, OHLCV_COLUMNS
from src.validation.validate_ohlcv import validate_ohlcv, log_issues

_YFINANCE_COLUMN_MAP: dict[str, str] = {
    "Date": "timestamp",
    "Datetime": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class HistoricalFetcher:
    """Fetch historical market data from Yahoo Finance and store it locally."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize the fetcher with a database manager instance.

        Args:
            db_manager: Database manager used to persist OHLCV candles.
        """
        self.db_manager = db_manager

    @staticmethod
    def _to_yfinance_symbol(symbol: str) -> str:
        """Return the Yahoo Finance ticker, appending ``.NS`` for NSE equities."""
        if symbol.startswith("^") or symbol.endswith(".NS"):
            return symbol
        return f"{symbol}.NS"

    @staticmethod
    def _to_storage_symbol(symbol: str) -> str:
        """Return the canonical symbol stored in the database (without ``.NS``)."""
        if symbol.endswith(".NS"):
            return symbol.removesuffix(".NS")
        return symbol

    @staticmethod
    def _prepare_candles(
        raw_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Transform a yfinance history DataFrame into the DB schema."""
        candles = raw_df.reset_index().rename(columns=_YFINANCE_COLUMN_MAP)
        candles = candles.loc[
            :, ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
        ]
        candles["symbol"] = symbol
        candles["timeframe"] = timeframe
        candles["volume"] = candles["volume"].fillna(0).astype("int64")
        return candles.loc[:, OHLCV_COLUMNS]

    def fetch_and_store(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "1y",
    ) -> None:
        """Download historical candles and persist them to DuckDB.

        Args:
            symbol: Equity ticker (for example, ``RELIANCE`` or ``RELIANCE.NS``).
            interval: Candle interval passed to yfinance (for example, ``1d``).
            period: Lookback period passed to yfinance (for example, ``1y``).
        """
        yfinance_symbol = self._to_yfinance_symbol(symbol)
        storage_symbol = self._to_storage_symbol(symbol)

        print(
            f"Fetching {storage_symbol} ({yfinance_symbol}) "
            f"[interval={interval}, period={period}]..."
        )

        raw_df = yf.Ticker(yfinance_symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )

        if raw_df.empty:
            print(f"No data returned for {yfinance_symbol}.")
            return

        candles = self._prepare_candles(raw_df, storage_symbol, interval)
        row_count = len(candles)

        print(f"Fetched {row_count} rows for {storage_symbol}.")

        # Validate data quality before storage
        issues_df = validate_ohlcv(candles, storage_symbol)
        if not issues_df.empty:
            print(f"Data quality issues detected for {storage_symbol}: {len(issues_df)} issues.")
            with self.db_manager.get_connection() as conn:
                log_issues(conn, issues_df)
        
        self.db_manager.save_candles(candles, timeframe=interval)
        print(f"Saved {row_count} rows for {storage_symbol} (timeframe={interval}).")

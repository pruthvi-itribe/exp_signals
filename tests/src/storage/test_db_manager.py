"""Tests for src/storage/db_manager.py: DatabaseManager's connection
lifecycle and the on-disk save/fetch round trip it wraps around upsert_ohlcv.

Would catch: init not creating the schema/parent directory, save_candles not
actually persisting, fetch_candles returning rows out of order or crossing
symbols/timeframes, or the upsert contract breaking when driven through this
class specifically (as opposed to calling upsert_ohlcv directly).

IMPORTANT: never instantiate DatabaseManager() with no args in a test -- that
touches the real project database at data/trading_data.duckdb. Always pass
an explicit tmp_path-based db_path.
"""

import pandas as pd
import pytest

from src.storage.db_manager import DatabaseManager, OHLCV_COLUMNS


def _row(symbol, timestamp, open_, high, low, close, adj_close, volume):
    return {
        "symbol": symbol,
        "timestamp": pd.Timestamp(timestamp),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": adj_close,
        "volume": volume,
    }


def test_init_creates_db_file_and_schema(tmp_path):
    db_path = tmp_path / "sub" / "test.duckdb"
    manager = DatabaseManager(db_path=db_path)

    assert db_path.exists()
    with manager.get_connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info('ohlcv_data')").fetchall()}
    for col in OHLCV_COLUMNS:
        assert col in columns
    assert "fetched_at" in columns


def test_save_and_fetch_candles_roundtrip_sorted_ascending(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    df = pd.DataFrame(
        [
            _row("AAA", "2024-01-02", 101, 106, 100, 105, 105, 1200),
            _row("AAA", "2024-01-01", 100, 105, 99, 104, 104, 1000),
        ]
    )
    manager.save_candles(df, timeframe="1d")

    fetched = manager.fetch_candles("AAA", "1d")
    assert list(fetched["timestamp"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert fetched.iloc[0]["close"] == pytest.approx(104.0)
    assert fetched.iloc[1]["close"] == pytest.approx(105.0)


def test_save_candles_empty_df_is_noop(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    manager.save_candles(pd.DataFrame(), timeframe="1d")
    assert manager.fetch_candles("ANY", "1d").empty


def test_fetch_candles_unknown_symbol_returns_empty_df(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    manager.save_candles(
        pd.DataFrame([_row("AAA", "2024-01-01", 100, 105, 99, 104, 104, 1000)]), timeframe="1d"
    )
    assert manager.fetch_candles("NOPE", "1d").empty


def test_save_candles_upsert_replaces_existing_row_not_duplicates(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    manager.save_candles(
        pd.DataFrame([_row("AAA", "2024-01-01", 100, 105, 99, 104, 104, 1000)]), timeframe="1d"
    )
    manager.save_candles(
        pd.DataFrame([_row("AAA", "2024-01-01", 100, 108, 99, 107, 107, 1500)]), timeframe="1d"
    )

    fetched = manager.fetch_candles("AAA", "1d")
    assert len(fetched) == 1
    assert fetched.iloc[0]["close"] == pytest.approx(107.0)
    assert fetched.iloc[0]["volume"] == 1500


def test_fetch_candles_does_not_cross_timeframes(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    manager.save_candles(
        pd.DataFrame([_row("AAA", "2024-01-01", 100, 105, 99, 104, 104, 1000)]), timeframe="1d"
    )
    manager.save_candles(
        pd.DataFrame([_row("AAA", "2024-01-01", 100, 101, 99, 100, 100, 10)]), timeframe="1h"
    )

    daily = manager.fetch_candles("AAA", "1d")
    hourly = manager.fetch_candles("AAA", "1h")
    assert len(daily) == 1 and daily.iloc[0]["close"] == pytest.approx(104.0)
    assert len(hourly) == 1 and hourly.iloc[0]["close"] == pytest.approx(100.0)


def test_get_connection_closes_on_exit(tmp_path):
    manager = DatabaseManager(db_path=tmp_path / "test.duckdb")
    with manager.get_connection() as conn:
        conn.execute("SELECT 1").fetchone()
    with pytest.raises(Exception):
        conn.execute("SELECT 1")  # connection was closed on context exit

"""Tests for src/storage/upsert.py: the core (symbol, timestamp, timeframe)
upsert contract used by every OHLCV ingestion path in this project.

Would catch: a broken ON CONFLICT clause that inserts duplicates instead of
updating, an incorrect rows_inserted/rows_updated split in the returned
summary, a missing-column input silently proceeding instead of raising, or
the primary key being narrowed so two different timeframes at the same
symbol/timestamp collide.
"""

import duckdb
import pandas as pd
import pytest

from src.storage.upsert import _prepare_staging_dataframe, ensure_ohlcv_upsert_schema, upsert_ohlcv


def _df(rows):
    return pd.DataFrame(rows)


def test_ensure_ohlcv_upsert_schema_is_idempotent():
    conn = duckdb.connect(":memory:")
    ensure_ohlcv_upsert_schema(conn)
    ensure_ohlcv_upsert_schema(conn)  # must not error the second time


def test_upsert_ohlcv_empty_df_returns_zero_summary():
    conn = duckdb.connect(":memory:")
    summary = upsert_ohlcv(conn, pd.DataFrame(), timeframe="1d")
    assert summary == {"rows_inserted": 0, "rows_updated": 0, "total_rows": 0}


def test_prepare_staging_dataframe_missing_columns_raises():
    df = _df([{"symbol": "AAA", "timestamp": "2024-01-01"}])  # missing OHLC/adj_close/volume
    with pytest.raises(ValueError, match="open"):
        _prepare_staging_dataframe(df, "1d")


def test_upsert_ohlcv_first_insert_all_rows_new():
    conn = duckdb.connect(":memory:")
    df = _df(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.0, "high": 105.0,
             "low": 99.0, "close": 104.0, "adj_close": 104.0, "volume": 1000},
            {"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-02"), "open": 101.0, "high": 106.0,
             "low": 100.0, "close": 105.0, "adj_close": 105.0, "volume": 1200},
        ]
    )
    summary = upsert_ohlcv(conn, df, timeframe="1d")
    assert summary == {"rows_inserted": 2, "rows_updated": 0, "total_rows": 2}
    assert conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0] == 2


def test_upsert_ohlcv_conflict_updates_in_place_not_duplicated():
    conn = duckdb.connect(":memory:")
    initial = _df(
        [{"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.0, "high": 105.0,
          "low": 99.0, "close": 104.0, "adj_close": 104.0, "volume": 1000}]
    )
    upsert_ohlcv(conn, initial, timeframe="1d")

    refresh = _df(
        [{"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.5, "high": 107.0,
          "low": 100.0, "close": 106.5, "adj_close": 106.5, "volume": 1500}]
    )
    summary = upsert_ohlcv(conn, refresh, timeframe="1d")

    assert summary == {"rows_inserted": 0, "rows_updated": 1, "total_rows": 1}
    rows = conn.execute("SELECT * FROM ohlcv_data WHERE symbol='AAA'").df()
    assert len(rows) == 1  # replaced, not duplicated
    assert rows.iloc[0]["close"] == pytest.approx(106.5)
    assert rows.iloc[0]["volume"] == 1500


def test_upsert_ohlcv_mixed_insert_and_update_in_one_batch():
    conn = duckdb.connect(":memory:")
    upsert_ohlcv(
        conn,
        _df([{"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.0, "high": 105.0,
              "low": 99.0, "close": 104.0, "adj_close": 104.0, "volume": 1000}]),
        timeframe="1d",
    )

    batch = _df(
        [
            {"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.5, "high": 107.0,
             "low": 100.0, "close": 106.5, "adj_close": 106.5, "volume": 1500},  # update
            {"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-02"), "open": 102.0, "high": 108.0,
             "low": 101.0, "close": 107.0, "adj_close": 107.0, "volume": 1300},  # new
        ]
    )
    summary = upsert_ohlcv(conn, batch, timeframe="1d")
    assert summary == {"rows_inserted": 1, "rows_updated": 1, "total_rows": 2}
    assert conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0] == 2


def test_upsert_ohlcv_different_timeframe_same_key_is_not_a_conflict():
    """symbol+timestamp alone must not collide across timeframes -- the PK
    is (symbol, timestamp, timeframe). Would catch the PK being narrowed to
    (symbol, timestamp) by accident."""
    conn = duckdb.connect(":memory:")
    row = {"symbol": "AAA", "timestamp": pd.Timestamp("2024-01-01"), "open": 100.0, "high": 105.0,
           "low": 99.0, "close": 104.0, "adj_close": 104.0, "volume": 1000}
    upsert_ohlcv(conn, _df([row]), timeframe="1d")
    summary = upsert_ohlcv(conn, _df([row]), timeframe="1h")

    assert summary == {"rows_inserted": 1, "rows_updated": 0, "total_rows": 1}
    assert conn.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0] == 2

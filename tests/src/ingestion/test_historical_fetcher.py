"""Tests for src/ingestion/historical_fetcher.py's HistoricalFetcher.

All yfinance access is mocked -- no test here makes a real network call.
Where a scenario would depend on src/validation/validate_ohlcv.py's exact
check thresholds, validate_ohlcv is patched to a controlled stub instead
(that module has its own test suite elsewhere).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ingestion import historical_fetcher as hf_module
from src.ingestion.historical_fetcher import HistoricalFetcher
from src.storage.db_manager import OHLCV_COLUMNS


# --------------------------------------------------------------------------
# _to_yfinance_symbol / _to_storage_symbol
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("RELIANCE", "RELIANCE.NS"),   # bare storage symbol -> gets .NS appended
        ("RELIANCE.NS", "RELIANCE.NS"),  # already suffixed -> unchanged
        ("^NSEI", "^NSEI"),             # index ticker -> never gets .NS
    ],
)
def test_to_yfinance_symbol(symbol, expected):
    assert HistoricalFetcher._to_yfinance_symbol(symbol) == expected


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("RELIANCE.NS", "RELIANCE"),
        ("RELIANCE", "RELIANCE"),  # no suffix to strip -> unchanged
        ("^NSEI", "^NSEI"),
    ],
)
def test_to_storage_symbol(symbol, expected):
    assert HistoricalFetcher._to_storage_symbol(symbol) == expected


# --------------------------------------------------------------------------
# _prepare_candles
# --------------------------------------------------------------------------


def test_prepare_candles_maps_columns_fills_missing_volume_and_orders_output():
    """Would catch a wrong yfinance->storage column rename, a missing
    symbol/timeframe column, or NaN volume not being coerced to 0/int64."""
    raw_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [104.0, 105.0],
            "Adj Close": [104.0, 105.0],
            "Volume": [1000.0, float("nan")],
        },
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date"),
    )

    candles = HistoricalFetcher._prepare_candles(raw_df, "RELIANCE", "1d")

    assert list(candles.columns) == list(OHLCV_COLUMNS)
    assert (candles["symbol"] == "RELIANCE").all()
    assert (candles["timeframe"] == "1d").all()
    assert candles["volume"].tolist() == [1000, 0]
    assert candles["volume"].dtype == "int64"
    assert pd.Timestamp(candles["timestamp"].iloc[0]) == pd.Timestamp("2024-01-01")


# --------------------------------------------------------------------------
# fetch_and_store
# --------------------------------------------------------------------------


def _fake_ticker(history_df):
    ticker = MagicMock()
    ticker.history.return_value = history_df
    return ticker


def test_fetch_and_store_empty_history_does_not_save(monkeypatch):
    """An empty yfinance response (e.g. delisted/wrong ticker) must not call
    save_candles at all -- would catch a bug that saves an empty/garbage
    frame instead of returning early."""
    monkeypatch.setattr(hf_module.yf, "Ticker", lambda symbol: _fake_ticker(pd.DataFrame()))
    db_manager = MagicMock()

    HistoricalFetcher(db_manager).fetch_and_store("FAKE")

    db_manager.save_candles.assert_not_called()


def test_fetch_and_store_saves_clean_data(monkeypatch):
    """Happy path: a non-empty, issue-free fetch results in exactly one
    save_candles call with the prepared candles and the right timeframe."""
    raw_df = pd.DataFrame(
        {
            "Open": [100.0], "High": [105.0], "Low": [99.0],
            "Close": [104.0], "Adj Close": [104.0], "Volume": [1000.0],
        },
        index=pd.DatetimeIndex(["2024-01-01"], name="Date"),
    )
    monkeypatch.setattr(hf_module.yf, "Ticker", lambda symbol: _fake_ticker(raw_df))
    monkeypatch.setattr(hf_module, "validate_ohlcv", lambda candles, symbol: pd.DataFrame())
    db_manager = MagicMock()

    HistoricalFetcher(db_manager).fetch_and_store("RELIANCE", interval="1d")

    db_manager.save_candles.assert_called_once()
    saved_df, kwargs = db_manager.save_candles.call_args[0][0], db_manager.save_candles.call_args[1]
    assert len(saved_df) == 1
    assert kwargs["timeframe"] == "1d"


def test_fetch_and_store_logs_issues_but_still_saves(monkeypatch):
    """When validate_ohlcv reports issues, they're logged via
    db_manager.get_connection() -- and, per this module's actual (documented
    here, not asserted-as-correct) behavior, save_candles is STILL called
    unconditionally afterward: fetch_and_store does not filter out
    error-severity rows before storing, unlike src.universe's
    _fetch_validate_upsert_one which does. This is worth knowing since the
    two code paths look similar but differ here -- flagged for awareness,
    not as a bug: fetch_and_store's only call site (main.py) is currently
    commented out, so this asymmetry isn't live in the active fetch pipeline
    (src.universe.bulk_fetch_and_store / resumable_bulk_fetch)."""
    raw_df = pd.DataFrame(
        {
            "Open": [100.0], "High": [105.0], "Low": [99.0],
            "Close": [104.0], "Adj Close": [104.0], "Volume": [1000.0],
        },
        index=pd.DatetimeIndex(["2024-01-01"], name="Date"),
    )
    issues_df = pd.DataFrame(
        [{"symbol": "RELIANCE", "date": "2024-01-01", "issue_type": "test", "severity": "error", "details": ""}]
    )
    monkeypatch.setattr(hf_module.yf, "Ticker", lambda symbol: _fake_ticker(raw_df))
    monkeypatch.setattr(hf_module, "validate_ohlcv", lambda candles, symbol: issues_df)

    logged = {}
    monkeypatch.setattr(hf_module, "log_issues", lambda conn, df: logged.setdefault("issues", df))

    db_manager = MagicMock()
    fake_conn = MagicMock()
    db_manager.get_connection.return_value.__enter__.return_value = fake_conn

    HistoricalFetcher(db_manager).fetch_and_store("RELIANCE")

    assert logged["issues"] is issues_df
    db_manager.save_candles.assert_called_once()  # documented current behavior -- see docstring above

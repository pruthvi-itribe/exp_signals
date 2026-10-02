"""Shared fixtures and synthetic-data builders for the trading_bot test suite.

Import from here instead of re-defining an in-memory DB / OHLCV builder in
each test module. This is what makes the suite scalable: a new module's test
file seeds a connection with `make_conn()`, inserts rows with `insert_ohlcv`/
`insert_signal`, and asserts on hand-computed expected values -- the same
pattern every other module's tests use. If a new module needs a builder that
doesn't exist yet, add it here (following the existing style) rather than
duplicating it locally in one test file.

Nothing in this module hits the network or a real file on disk except
`REAL_DB_PATH`, which is only a path -- tests that want the real project
database must check `REAL_DB_PATH.exists()` themselves and `pytest.skip(...)`
if it's absent (see `tests/test_backtest.py` for the pattern), typically also
marked `@pytest.mark.realdata` so the reason shows up in `-v` output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd

from src.storage.upsert import ensure_ohlcv_upsert_schema
from src.strategy import ensure_signals_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"


def make_conn() -> duckdb.DuckDBPyConnection:
    """A fresh in-memory DuckDB connection with the production ohlcv_data/signals schemas."""
    conn = duckdb.connect(":memory:")
    ensure_ohlcv_upsert_schema(conn)
    ensure_signals_schema(conn)
    return conn


def insert_ohlcv(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    rows: Iterable[Sequence[object]],
    timeframe: str = "1d",
) -> None:
    """Insert OHLCV rows for one symbol into an ohlcv_data table.

    Each row is either:
      - (date_str, open, close)                       -- high/low derived
        from open/close, adj_close = close, volume = 0.
      - (date_str, open, high, low, close, volume)     -- full control, e.g.
        for OHLC-violation boundary tests where high/low/volume matter.
      - (date_str, open, high, low, close, adj_close, volume) -- full
        control including a divergent adj_close (corporate-action tests).

    fetched_at is set equal to the row's own date -- its exact value is
    never asserted on anywhere in this suite, it just needs to exist.
    """
    normalized = []
    for row in rows:
        if len(row) == 3:
            d, o, c = row
            h, l, adj, v = max(o, c), min(o, c), c, 0
        elif len(row) == 6:
            d, o, h, l, c, v = row
            adj = c
        elif len(row) == 7:
            d, o, h, l, c, adj, v = row
        else:
            raise ValueError(f"expected a 3-, 6-, or 7-tuple row, got {row!r}")
        normalized.append((symbol, d, o, h, l, c, adj, v, timeframe, d))

    conn.executemany(
        """
        INSERT INTO ohlcv_data
            (symbol, timestamp, open, high, low, close, adj_close, volume, timeframe, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        normalized,
    )


def insert_signal(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    date_str: str,
    strategy: str,
    signal_type: str,
    price: float | None = None,
    reason: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO signals (symbol, date, strategy, signal_type, price, reason)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [symbol, date_str, strategy, signal_type, price, reason],
    )


def synthetic_ohlcv_df(
    rows: Iterable[Sequence[object]],
    symbol: str = "TESTCO",
) -> pd.DataFrame:
    """Build a standalone OHLCV DataFrame (no DB involved) for testing pure
    functions that take a DataFrame directly -- indicators, strategies,
    validate_ohlcv, signal_library, etc.

    Each row is (date_str, open, high, low, close, volume); adj_close
    mirrors close. Returns columns: symbol, date, open, high, low, close,
    adj_close, volume, with `date` as a proper Timestamp column sorted
    ascending -- the shape every one of those modules' functions expects.

    For a module that only wants a subset of columns, slice the result
    (`df[["date", "open", "close"]]`) rather than adding a parallel builder.
    """
    records = [
        {
            "symbol": symbol,
            "date": pd.Timestamp(d),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "adj_close": c,
            "volume": v,
        }
        for d, o, h, l, c, v in rows
    ]
    return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)


def zero_cost(monkeypatch) -> None:
    """Patch backtest.calculate_transaction_cost to always return 0.

    Import backtest lazily to avoid pulling backtest.py's own imports
    (src.universe, src.strategy) into modules that don't otherwise need it.
    """
    import backtest

    monkeypatch.setattr(backtest, "calculate_transaction_cost", lambda trade_value, side: 0.0)


def seed_active_universe(
    conn: duckdb.DuckDBPyConnection,
    symbols: Iterable[str],
    index_name: str = "NIFTY50",
    added_date: str = "2020-01-01",
) -> None:
    """Mark `symbols` (storage symbols, no `.NS`) active in `universe` as of `added_date`.

    For tests of code that calls `src.universe.get_active_universe` (which
    requires at least one non-removed row per symbol) without needing the
    real fetch/sync pipeline.
    """
    from src.universe import ensure_universe_schema

    ensure_universe_schema(conn)
    conn.executemany(
        """
        INSERT INTO universe (symbol, yf_ticker, index_name, added_date, removed_date, is_active)
        VALUES (?, ?, ?, ?, NULL, true)
        """,
        [(symbol, f"{symbol}.NS", index_name, added_date) for symbol in symbols],
    )

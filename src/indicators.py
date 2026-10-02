"""
Technical indicator computation and storage for daily OHLCV data.

Computes per-symbol rolling features from ``adj_close`` and persists them in
``indicators_daily``, decoupled from raw OHLCV so multiple strategies can reuse
the same feature set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.universe import get_active_universe

DEFAULT_DB_PATH: Path = Path("data/trading_data.duckdb")
DAILY_TIMEFRAME: str = "1d"

OHLCV_INPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

BB_WINDOW: int = 20
BB_NUM_STD: float = 2.0

INDICATOR_OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "daily_return",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "volatility_20",
    "bb_middle",
    "bb_std",
    "bb_upper",
    "bb_lower",
)

INDICATOR_VALUE_COLUMNS: tuple[str, ...] = INDICATOR_OUTPUT_COLUMNS[2:]

StoreSummary = dict[str, int]


def ensure_indicators_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``indicators_daily`` table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indicators_daily (
            symbol         VARCHAR NOT NULL,
            date           DATE NOT NULL,
            daily_return   DOUBLE,
            sma_20         DOUBLE,
            sma_50         DOUBLE,
            ema_12         DOUBLE,
            ema_26         DOUBLE,
            rsi_14         DOUBLE,
            volatility_20  DOUBLE,
            bb_middle      DOUBLE,
            bb_std         DOUBLE,
            bb_upper       DOUBLE,
            bb_lower       DOUBLE,
            computed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    # Table may already exist from before Bollinger columns were added.
    conn.execute("ALTER TABLE indicators_daily ADD COLUMN IF NOT EXISTS bb_middle DOUBLE")
    conn.execute("ALTER TABLE indicators_daily ADD COLUMN IF NOT EXISTS bb_std DOUBLE")
    conn.execute("ALTER TABLE indicators_daily ADD COLUMN IF NOT EXISTS bb_upper DOUBLE")
    conn.execute("ALTER TABLE indicators_daily ADD COLUMN IF NOT EXISTS bb_lower DOUBLE")


def _normalize_ohlcv_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV input to expected column names and sort order."""
    if df.empty:
        return df.copy()

    working = df.copy()
    if "date" not in working.columns and "timestamp" in working.columns:
        working["date"] = pd.to_datetime(working["timestamp"], errors="coerce").dt.normalize()

    missing = [col for col in OHLCV_INPUT_COLUMNS if col not in working.columns]
    if missing:
        raise ValueError(f"OHLCV DataFrame is missing required columns: {', '.join(missing)}")

    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.sort_values(["symbol", "date"]).reset_index(drop=True)
    return working.loc[:, OHLCV_INPUT_COLUMNS]


def _compute_rsi_14(adj_close: pd.Series) -> pd.Series:
    """Compute 14-period RSI from ``adj_close`` using explicit average gain/loss.

    Formula (per bar, after 14 prior changes exist):
        change = adj_close - adj_close.shift(1)
        gain   = max(change, 0)
        loss   = max(-change, 0)
        avg_gain = mean(gain over last 14 bars)
        avg_loss = mean(loss over last 14 bars)
        RS  = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))

    RSI > 70 is often interpreted as overbought; RSI < 30 as oversold.
    """
    change = adj_close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain.notna() & avg_loss.notna())
    return rsi


def _compute_symbol_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicator columns for a single symbol, sorted by date."""
    price = group["adj_close"]

    # Daily return: simple pct change; base input for volatility and risk metrics.
    daily_return = price.pct_change()

    # SMA: trend direction and dynamic support/resistance levels.
    sma_20 = price.rolling(window=20, min_periods=20).mean()
    sma_50 = price.rolling(window=50, min_periods=50).mean()

    # EMA: responsive trend averages; ema_12/ema_26 are MACD building blocks.
    ema_12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = price.ewm(span=26, adjust=False, min_periods=26).mean()

    rsi_14 = _compute_rsi_14(price)

    # Rolling return volatility: recent risk / regime-change signal.
    volatility_20 = daily_return.rolling(window=20, min_periods=20).std()

    # Bollinger Bands: bb_middle reuses sma_20 (same 20-day window on adj_close),
    # stored under its own column name for clarity as a band rather than a trend MA.
    bb_middle = sma_20
    bb_std = price.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std()
    bb_upper = bb_middle + BB_NUM_STD * bb_std
    bb_lower = bb_middle - BB_NUM_STD * bb_std

    return pd.DataFrame(
        {
            "symbol": group["symbol"].values,
            "date": group["date"].values,
            "daily_return": daily_return.values,
            "sma_20": sma_20.values,
            "sma_50": sma_50.values,
            "ema_12": ema_12.values,
            "ema_26": ema_26.values,
            "rsi_14": rsi_14.values,
            "volatility_20": volatility_20.values,
            "bb_middle": bb_middle.values,
            "bb_std": bb_std.values,
            "bb_upper": bb_upper.values,
            "bb_lower": bb_lower.values,
        }
    )


def _print_warmup_summary(indicators: pd.DataFrame) -> None:
    """Print how many rows per symbol still have at least one NaN indicator."""
    if indicators.empty:
        print("Warm-up summary: no indicator rows computed.")
        return

    nan_mask = indicators.loc[:, INDICATOR_VALUE_COLUMNS].isna().any(axis=1)
    warmup_counts = (
        indicators.loc[nan_mask]
        .groupby("symbol", sort=True)
        .size()
        .rename("rows_with_nan_indicator")
    )

    print("\nWarm-up summary (rows with at least one NaN indicator per symbol):")
    if warmup_counts.empty:
        print("  None — all rows fully populated.")
    else:
        for symbol, count in warmup_counts.items():
            print(f"  {symbol}: {count}")
        print(
            f"  Expected ~50 rows per symbol (largest lookback: sma_50 / ema_26); "
            f"median={warmup_counts.median():.0f}, max={warmup_counts.max()}."
        )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily technical indicators for one or many symbols.

    All rolling and EWM calculations are performed per ``symbol`` group on
    ``adj_close``, sorted by ``date``, so windows never leak across stocks.

    Args:
        df: OHLCV DataFrame with columns ``symbol``, ``date`` (or ``timestamp``),
            ``open``, ``high``, ``low``, ``close``, ``adj_close``, ``volume``.

    Returns:
        DataFrame with ``symbol``, ``date``, and indicator columns. Warm-up rows
        retain NaN until each indicator's lookback is satisfied.
    """
    ohlcv = _normalize_ohlcv_input(df)
    if ohlcv.empty:
        return pd.DataFrame(columns=list(INDICATOR_OUTPUT_COLUMNS))

    indicator_frames = [
        _compute_symbol_indicators(group)
        for _, group in ohlcv.groupby("symbol", sort=True)
    ]
    indicators = pd.concat(indicator_frames, ignore_index=True)
    indicators = indicators.loc[:, INDICATOR_OUTPUT_COLUMNS]

    _print_warmup_summary(indicators)
    return indicators


def _prepare_indicator_staging(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``computed_at`` and validate indicator staging columns."""
    missing = [col for col in INDICATOR_OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Indicator DataFrame is missing required columns: {', '.join(missing)}")

    staging = df.loc[:, INDICATOR_OUTPUT_COLUMNS].copy()
    staging["date"] = pd.to_datetime(staging["date"], errors="coerce").dt.normalize()
    staging["computed_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return staging


def store_indicators(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> StoreSummary:
    """Upsert computed indicators into ``indicators_daily``.

    Uses the same insert-on-conflict-update pattern as ``upsert_ohlcv``.

    Args:
        conn: Open DuckDB connection.
        df: Indicator DataFrame from ``compute_indicators``.

    Returns:
        Summary with ``rows_inserted``, ``rows_updated``, and ``total_rows``.
    """
    empty_summary: StoreSummary = {
        "rows_inserted": 0,
        "rows_updated": 0,
        "total_rows": 0,
    }
    if df.empty:
        return empty_summary

    ensure_indicators_schema(conn)
    staging = _prepare_indicator_staging(df)
    total_rows = len(staging)

    conn.register("_indicator_staging", staging)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            rows_updated = conn.execute(
                """
                SELECT COUNT(*)::BIGINT
                FROM _indicator_staging AS staging
                INNER JOIN indicators_daily AS existing
                    ON staging.symbol = existing.symbol
                   AND staging.date = existing.date
                """
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO indicators_daily (
                    symbol,
                    date,
                    daily_return,
                    sma_20,
                    sma_50,
                    ema_12,
                    ema_26,
                    rsi_14,
                    volatility_20,
                    bb_middle,
                    bb_std,
                    bb_upper,
                    bb_lower,
                    computed_at
                )
                SELECT
                    symbol,
                    date,
                    daily_return,
                    sma_20,
                    sma_50,
                    ema_12,
                    ema_26,
                    rsi_14,
                    volatility_20,
                    bb_middle,
                    bb_std,
                    bb_upper,
                    bb_lower,
                    computed_at
                FROM _indicator_staging
                ON CONFLICT (symbol, date) DO UPDATE SET
                    daily_return = excluded.daily_return,
                    sma_20 = excluded.sma_20,
                    sma_50 = excluded.sma_50,
                    ema_12 = excluded.ema_12,
                    ema_26 = excluded.ema_26,
                    rsi_14 = excluded.rsi_14,
                    volatility_20 = excluded.volatility_20,
                    bb_middle = excluded.bb_middle,
                    bb_std = excluded.bb_std,
                    bb_upper = excluded.bb_upper,
                    bb_lower = excluded.bb_lower,
                    computed_at = excluded.computed_at
                """
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.unregister("_indicator_staging")

    rows_inserted = total_rows - int(rows_updated)
    return {
        "rows_inserted": rows_inserted,
        "rows_updated": int(rows_updated),
        "total_rows": total_rows,
    }


def _load_daily_ohlcv(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV rows from ``ohlcv_data`` (timeframe ``1d``)."""
    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        query = f"""
            SELECT
                symbol,
                timestamp::DATE AS date,
                open,
                high,
                low,
                close,
                adj_close,
                volume
            FROM ohlcv_data
            WHERE timeframe = ?
              AND symbol IN ({placeholders})
            ORDER BY symbol, date
        """
        params: list[object] = [DAILY_TIMEFRAME, *symbols]
    else:
        query = """
            SELECT
                symbol,
                timestamp::DATE AS date,
                open,
                high,
                low,
                close,
                adj_close,
                volume
            FROM ohlcv_data
            WHERE timeframe = ?
            ORDER BY symbol, date
        """
        params = [DAILY_TIMEFRAME]

    return conn.execute(query, params).df()


def _resolve_symbols(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None,
) -> list[str]:
    """Return storage symbols (no ``.NS`` suffix) for indicator computation."""
    if symbols is not None:
        return [
            symbol.removesuffix(".NS") if symbol.endswith(".NS") else symbol
            for symbol in symbols
        ]

    yf_tickers = get_active_universe(conn)
    return [ticker.removesuffix(".NS") for ticker in yf_tickers]


def compute_and_store_all(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
) -> dict[str, object]:
    """Load OHLCV, compute indicators, and upsert into ``indicators_daily``.

    Reads daily bars from ``ohlcv_data`` (``timeframe = '1d'``). When ``symbols``
    is omitted, uses all currently active universe symbols.

    Safe to re-run after adding more historical OHLCV — existing rows are updated
    on conflict.

    Args:
        conn: Open DuckDB connection.
        symbols: Optional list of storage symbols to restrict computation.

    Returns:
        Combined summary with symbol count, OHLCV rows loaded, and upsert stats.
    """
    target_symbols = _resolve_symbols(conn, symbols)
    ohlcv = _load_daily_ohlcv(conn, target_symbols)

    if ohlcv.empty:
        return {
            "symbols": 0,
            "ohlcv_rows": 0,
            "rows_inserted": 0,
            "rows_updated": 0,
            "total_rows": 0,
        }

    indicators = compute_indicators(ohlcv)
    store_summary = store_indicators(conn, indicators)

    return {
        "symbols": ohlcv["symbol"].nunique(),
        "ohlcv_rows": len(ohlcv),
        **store_summary,
    }


if __name__ == "__main__":
    PREVIEW_SYMBOL = "RELIANCE"

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        ohlcv = _load_daily_ohlcv(conn)
        print(f"Loaded {len(ohlcv)} daily OHLCV rows for {ohlcv['symbol'].nunique()} symbols.")

        indicators = compute_indicators(ohlcv)

        preview = (
            indicators.loc[indicators["symbol"] == PREVIEW_SYMBOL]
            .sort_values("date")
            .tail(10)
        )
        print(f"\n--- Preview: last 10 indicator rows for {PREVIEW_SYMBOL} ---")
        print(
            preview.to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}" if pd.notna(value) else "NaN",
            )
        )

        store_summary = store_indicators(conn, indicators)
        print("\n--- Store summary ---")
        print(f"Rows inserted: {store_summary['rows_inserted']}")
        print(f"Rows updated:  {store_summary['rows_updated']}")
        print(f"Total rows:    {store_summary['total_rows']}")

        stored_count = conn.execute(
            "SELECT COUNT(*) FROM indicators_daily"
        ).fetchone()[0]
        symbol_count = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM indicators_daily"
        ).fetchone()[0]
        print(f"\nindicators_daily now holds {stored_count} rows across {symbol_count} symbols.")
    finally:
        conn.close()

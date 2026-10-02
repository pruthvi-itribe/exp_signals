"""Forward returns: realized N-trading-day-ahead price moves, cached for signal screening.

Computed once per symbol/date and stored in ``forward_returns`` so
``research.screen`` doesn't recompute the same returns on every screening
run — this is a shared, read-heavy cache, not per-signal or per-strategy
state.

Note on naming: the project's daily OHLCV table is ``ohlcv_data`` (filtered
to ``timeframe = '1d'``), not ``ohlcv_daily`` — this module reads from the
former to match every other module in the codebase.
"""

from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from src.universe import DEFAULT_DB_PATH, get_active_universe

DAILY_TIMEFRAME: str = "1d"
DEFAULT_HORIZONS: list[int] = [1, 5, 10, 20, 40, 60]
FORWARD_RETURN_COLUMNS: tuple[str, ...] = tuple(f"fwd_return_{h}d" for h in DEFAULT_HORIZONS)

StoreSummary = dict[str, int]


def ensure_forward_returns_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``forward_returns`` table if it does not already exist.

    ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` also runs every time (same
    pattern as ``src.storage.upsert.ensure_ohlcv_upsert_schema``) so a table
    created before ``fwd_return_40d``/``fwd_return_60d`` existed gets them
    added in place, rather than needing a fresh table.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forward_returns (
            symbol         VARCHAR NOT NULL,
            date           DATE NOT NULL,
            fwd_return_1d  DOUBLE,
            fwd_return_5d  DOUBLE,
            fwd_return_10d DOUBLE,
            fwd_return_20d DOUBLE,
            computed_at    TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute("ALTER TABLE forward_returns ADD COLUMN IF NOT EXISTS fwd_return_40d DOUBLE")
    conn.execute("ALTER TABLE forward_returns ADD COLUMN IF NOT EXISTS fwd_return_60d DOUBLE")


def compute_forward_returns(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Compute forward returns per symbol using actual trading-day offsets.

    For each symbol (grouped, sorted by date), ``fwd_return_{h}d`` on date T
    is ``adj_close[row T+h] / adj_close[row T] - 1``, where "T+h" means h
    rows ahead in that symbol's own trading-day sequence (via
    ``Series.shift(-h)``) — not T plus h *calendar* days, which would land
    on the wrong row across weekends and holidays.

    The last ``h`` rows per symbol are NaN for ``fwd_return_{h}d`` (no
    future data exists yet to compute them) — left as NaN, not dropped, so
    every (symbol, date) row stays represented even when some horizons
    aren't yet resolvable for it.

    Args:
        df: OHLCV data with ``symbol``, ``date``, ``adj_close`` columns.
            Should include a symbol's full available history (not just a
            screening window) so later horizons aren't truncated.
        horizons: Trading-day horizons to compute. Defaults to
            ``DEFAULT_HORIZONS`` (``[1, 5, 10, 20, 40, 60]``) — the columns
            ``forward_returns`` has room for. Other horizons can be
            computed here for ad-hoc/in-memory use, but ``store_forward_returns``
            can only persist the six default ones.

    Returns:
        DataFrame with ``symbol``, ``date``, and one ``fwd_return_{h}d``
        column per horizon, sorted by symbol then date.
    """
    horizons = horizons if horizons is not None else DEFAULT_HORIZONS

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.sort_values(["symbol", "date"]).reset_index(drop=True)

    output_columns = ["symbol", "date", *[f"fwd_return_{h}d" for h in horizons]]
    if working.empty:
        return pd.DataFrame(columns=output_columns)

    frames: list[pd.DataFrame] = []
    for symbol, group in working.groupby("symbol", sort=True):
        price = group["adj_close"]
        result: dict[str, object] = {"symbol": group["symbol"].values, "date": group["date"].values}
        for horizon in horizons:
            future_price = price.shift(-horizon)
            result[f"fwd_return_{horizon}d"] = (future_price / price - 1).values
        frames.append(pd.DataFrame(result))

    return pd.concat(frames, ignore_index=True).loc[:, output_columns]


def store_forward_returns(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> StoreSummary:
    """Upsert computed forward returns into ``forward_returns``.

    Uses the same insert-on-conflict-update pattern as ``indicators.store_indicators``.
    ``df`` may omit some of the six ``fwd_return_*d`` columns (stored as
    NULL) but must not contain any ``fwd_return_*d`` column outside
    ``FORWARD_RETURN_COLUMNS`` — the table has no room for arbitrary horizons.

    Args:
        conn: Open DuckDB connection.
        df: Forward-returns DataFrame from ``compute_forward_returns``.

    Returns:
        Summary with ``rows_inserted``, ``rows_updated``, and ``total_rows``.
    """
    empty_summary: StoreSummary = {"rows_inserted": 0, "rows_updated": 0, "total_rows": 0}
    if df.empty:
        return empty_summary

    unsupported = sorted(
        col for col in df.columns
        if col.startswith("fwd_return_") and col not in FORWARD_RETURN_COLUMNS
    )
    if unsupported:
        raise ValueError(
            f"forward_returns only stores {FORWARD_RETURN_COLUMNS}; got unsupported columns: {unsupported}"
        )

    ensure_forward_returns_schema(conn)

    staging = df.copy()
    staging["date"] = pd.to_datetime(staging["date"], errors="coerce").dt.normalize()
    for column in FORWARD_RETURN_COLUMNS:
        if column not in staging.columns:
            staging[column] = pd.NA
    staging["computed_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    staging = staging.loc[:, ["symbol", "date", *FORWARD_RETURN_COLUMNS, "computed_at"]]
    total_rows = len(staging)

    column_list = ", ".join(FORWARD_RETURN_COLUMNS)
    update_list = ", ".join(f"{col} = excluded.{col}" for col in FORWARD_RETURN_COLUMNS)

    conn.register("_forward_returns_staging", staging)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            rows_updated = conn.execute(
                """
                SELECT COUNT(*)::BIGINT
                FROM _forward_returns_staging AS staging
                INNER JOIN forward_returns AS existing
                    ON staging.symbol = existing.symbol
                   AND staging.date = existing.date
                """
            ).fetchone()[0]

            conn.execute(
                f"""
                INSERT INTO forward_returns (symbol, date, {column_list}, computed_at)
                SELECT symbol, date, {column_list}, computed_at
                FROM _forward_returns_staging
                ON CONFLICT (symbol, date) DO UPDATE SET
                    {update_list},
                    computed_at = excluded.computed_at
                """
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.unregister("_forward_returns_staging")

    rows_inserted = total_rows - int(rows_updated)
    return {"rows_inserted": rows_inserted, "rows_updated": int(rows_updated), "total_rows": total_rows}


def _load_daily_adj_close(conn: duckdb.DuckDBPyConnection, symbols: list[str]) -> pd.DataFrame:
    """Load full-history ``symbol, date, adj_close`` rows from ``ohlcv_data`` for ``symbols``.

    No date bound: forward returns need each symbol's complete available
    series so ``compute_forward_returns``'s forward shift can resolve as
    many rows as possible before running out of future data.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "date", "adj_close"])

    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, timestamp::DATE AS date, adj_close
        FROM ohlcv_data
        WHERE timeframe = ?
          AND symbol IN ({placeholders})
        ORDER BY symbol, date
    """
    df = conn.execute(query, [DAILY_TIMEFRAME, *symbols]).df()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def _resolve_symbols(conn: duckdb.DuckDBPyConnection, symbols: list[str] | None) -> list[str]:
    """Return storage symbols (no ``.NS`` suffix), defaulting to the active universe."""
    if symbols is not None:
        return [s.removesuffix(".NS") if s.endswith(".NS") else s for s in symbols]
    return [t.removesuffix(".NS") for t in get_active_universe(conn)]


def compute_and_store_forward_returns(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    """Load OHLCV, compute forward returns, and upsert into ``forward_returns``.

    Mirrors ``indicators.compute_and_store_all``'s shape. Safe to re-run
    after new OHLCV data lands — existing rows are updated on conflict.

    Args:
        conn: Open DuckDB connection.
        symbols: Optional list of storage symbols. Defaults to the active universe.
        horizons: Horizons to compute and store. Defaults to ``DEFAULT_HORIZONS``.

    Returns:
        Combined summary with symbol count, OHLCV rows loaded, and upsert stats.
    """
    target_symbols = _resolve_symbols(conn, symbols)
    ohlcv = _load_daily_adj_close(conn, target_symbols)

    if ohlcv.empty:
        return {"symbols": 0, "ohlcv_rows": 0, "rows_inserted": 0, "rows_updated": 0, "total_rows": 0}

    forward_returns = compute_forward_returns(ohlcv, horizons=horizons)
    store_summary = store_forward_returns(conn, forward_returns)

    return {
        "symbols": ohlcv["symbol"].nunique(),
        "ohlcv_rows": len(ohlcv),
        **store_summary,
    }


if __name__ == "__main__":
    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        summary = compute_and_store_forward_returns(conn)
        print(f"Computed forward returns for {summary['symbols']} symbols ({summary['ohlcv_rows']} OHLCV rows).")
        print(f"Rows inserted: {summary['rows_inserted']}  |  Rows updated: {summary['rows_updated']}")
    finally:
        conn.close()

"""
Strategy orchestration: loading merged market data for strategies to consume,
running a strategy, and persisting/summarizing its signals.

Strategy classes themselves live in the ``strategies`` package (see
``strategies.base.Strategy``, ``strategies.registry``) — this module wires
one up to the database. It works with any ``Strategy`` subclass generically;
it has no knowledge of which concrete strategy it's driving.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy
from src.market_regime import attach_market_regime, load_market_regime
from src.universe import get_active_universe

DEFAULT_DB_PATH: Path = Path("data/trading_data.duckdb")
DAILY_TIMEFRAME: str = "1d"

StoreSignalsSummary = dict[str, int]


def ensure_signals_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``signals`` table if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            symbol        VARCHAR NOT NULL,
            date          DATE NOT NULL,
            strategy      VARCHAR NOT NULL,
            signal_type   VARCHAR NOT NULL,
            price         DOUBLE,
            reason        VARCHAR,
            generated_at  TIMESTAMP DEFAULT current_timestamp,
            PRIMARY KEY (symbol, date, strategy)
        )
        """
    )


def _resolve_symbols(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None,
) -> list[str]:
    """Return storage symbols (no ``.NS`` suffix) for strategy input loading."""
    if symbols is not None:
        return [
            symbol.removesuffix(".NS") if symbol.endswith(".NS") else symbol
            for symbol in symbols
        ]

    yf_tickers = get_active_universe(conn)
    return [ticker.removesuffix(".NS") for ticker in yf_tickers]


def load_strategy_input(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Load merged OHLCV, indicator, and market-regime data for strategy evaluation.

    Joins daily OHLCV from ``ohlcv_data`` (``timeframe = '1d'``) with
    ``indicators_daily`` on ``(symbol, date)``, then broadcasts the Nifty 50
    index's own regime state (``nifty_regime_bullish``,
    ``nifty_regime_breakdown`` -- see ``src.market_regime``) onto every row
    by date. Every strategy gets these two extra columns whether or not it
    uses them; a strategy that doesn't declare them in ``required_columns``
    just ignores them.

    Args:
        conn: Open DuckDB connection.
        symbols: Optional list of storage symbols. Defaults to active universe.

    Returns:
        DataFrame with OHLCV, indicator, and market-regime columns, sorted
        by symbol and date.
    """
    target_symbols = _resolve_symbols(conn, symbols)
    if not target_symbols:
        return pd.DataFrame()

    placeholders = ", ".join("?" for _ in target_symbols)
    query = f"""
        SELECT
            o.symbol,
            o.timestamp::DATE AS date,
            o.open,
            o.high,
            o.low,
            o.close,
            o.adj_close,
            o.volume,
            i.daily_return,
            i.sma_20,
            i.sma_50,
            i.ema_12,
            i.ema_26,
            i.rsi_14,
            i.volatility_20
        FROM ohlcv_data AS o
        INNER JOIN indicators_daily AS i
            ON o.symbol = i.symbol
           AND o.timestamp::DATE = i.date
        WHERE o.timeframe = ?
          AND o.symbol IN ({placeholders})
        ORDER BY o.symbol, date
    """
    params: list[object] = [DAILY_TIMEFRAME, *target_symbols]
    merged = conn.execute(query, params).df()
    return attach_market_regime(merged, load_market_regime(conn))


def _prepare_signal_staging(df: pd.DataFrame) -> pd.DataFrame:
    """Validate signal rows and attach ``generated_at``."""
    missing = [col for col in SIGNAL_OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Signals DataFrame is missing required columns: {', '.join(missing)}")

    staging = df.loc[:, SIGNAL_OUTPUT_COLUMNS].copy()
    staging["date"] = pd.to_datetime(staging["date"], errors="coerce").dt.normalize()
    staging["generated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return staging


def store_signals(
    conn: duckdb.DuckDBPyConnection,
    signals_df: pd.DataFrame,
) -> StoreSignalsSummary:
    """Upsert signal rows into the ``signals`` table.

    Conflicts on ``(symbol, date, strategy)`` update signal fields and
    refresh ``generated_at``.

    Args:
        conn: Open DuckDB connection.
        signals_df: Signal DataFrame from a strategy's ``generate_signals``.

    Returns:
        Summary with ``signals_inserted``, ``buy_count``, and ``sell_count``.
    """
    empty_summary: StoreSignalsSummary = {
        "signals_inserted": 0,
        "buy_count": 0,
        "sell_count": 0,
    }
    if signals_df.empty:
        return empty_summary

    ensure_signals_schema(conn)
    staging = _prepare_signal_staging(signals_df)
    buy_count = int((staging["signal_type"] == "BUY").sum())
    sell_count = int((staging["signal_type"] == "SELL").sum())

    conn.register("_signal_staging", staging)
    try:
        conn.execute("BEGIN TRANSACTION")
        try:
            rows_updated = conn.execute(
                """
                SELECT COUNT(*)::BIGINT
                FROM _signal_staging AS staging
                INNER JOIN signals AS existing
                    ON staging.symbol = existing.symbol
                   AND staging.date = existing.date
                   AND staging.strategy = existing.strategy
                """
            ).fetchone()[0]

            conn.execute(
                """
                INSERT INTO signals (
                    symbol,
                    date,
                    strategy,
                    signal_type,
                    price,
                    reason,
                    generated_at
                )
                SELECT
                    symbol,
                    date,
                    strategy,
                    signal_type,
                    price,
                    reason,
                    generated_at
                FROM _signal_staging
                ON CONFLICT (symbol, date, strategy) DO UPDATE SET
                    signal_type = excluded.signal_type,
                    price = excluded.price,
                    reason = excluded.reason,
                    generated_at = excluded.generated_at
                """
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.unregister("_signal_staging")

    signals_inserted = len(staging) - int(rows_updated)
    return {
        "signals_inserted": signals_inserted,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


def run_strategy(
    conn: duckdb.DuckDBPyConnection,
    strategy: Strategy,
    symbols: list[str] | None = None,
) -> StoreSignalsSummary:
    """Load data, generate signals, persist them, and print a run summary.

    Args:
        conn: Open DuckDB connection.
        strategy: Strategy instance to evaluate.
        symbols: Optional symbol filter. Defaults to active universe.

    Returns:
        Store summary from ``store_signals``.
    """
    input_df = load_strategy_input(conn, symbols=symbols)
    if input_df.empty:
        print(f"No strategy input rows found for strategy '{strategy.name}'.")
        return {"signals_inserted": 0, "buy_count": 0, "sell_count": 0}

    signals_df = strategy.generate_signals(input_df)
    summary = store_signals(conn, signals_df)

    symbol_count = input_df["symbol"].nunique()
    print(f"\nStrategy run: {strategy.name}")
    print(f"  Input rows:   {len(input_df)} across {symbol_count} symbols")
    print(f"  Signals emitted: {len(signals_df)}")
    print(f"  Inserted:     {summary['signals_inserted']}")
    print(f"  BUY signals:  {summary['buy_count']}")
    print(f"  SELL signals: {summary['sell_count']}")

    return summary


def summarize_signals(conn: duckdb.DuckDBPyConnection, strategy_name: str) -> None:
    """Print a sanity-check summary of stored signals for one strategy.

    Args:
        conn: Open DuckDB connection.
        strategy_name: Value in the ``signals.strategy`` column.
    """
    ensure_signals_schema(conn)

    totals = conn.execute(
        """
        SELECT signal_type, COUNT(*)::BIGINT AS count
        FROM signals
        WHERE strategy = ?
        GROUP BY signal_type
        ORDER BY signal_type
        """,
        [strategy_name],
    ).df()

    print(f"\n=== Signal summary: {strategy_name} ===")

    if totals.empty:
        print("No signals stored for this strategy.")
        return

    buy_total = int(totals.loc[totals["signal_type"] == "BUY", "count"].sum())
    sell_total = int(totals.loc[totals["signal_type"] == "SELL", "count"].sum())
    print(f"Total BUY:  {buy_total}")
    print(f"Total SELL: {sell_total}")

    per_symbol = conn.execute(
        """
        SELECT symbol, signal_type, COUNT(*)::BIGINT AS count
        FROM signals
        WHERE strategy = ?
        GROUP BY symbol, signal_type
        ORDER BY symbol, signal_type
        """,
        [strategy_name],
    ).df()

    print("\nSignals per symbol:")
    if per_symbol.empty:
        print("  None")
    else:
        pivot = (
            per_symbol.pivot(index="symbol", columns="signal_type", values="count")
            .fillna(0)
            .astype(int)
        )
        for symbol, row in pivot.iterrows():
            buy_n = int(row.get("BUY", 0))
            sell_n = int(row.get("SELL", 0))
            print(f"  {symbol}: BUY={buy_n}, SELL={sell_n}")

    recent = conn.execute(
        """
        SELECT symbol, date, signal_type, price, reason
        FROM signals
        WHERE strategy = ?
        ORDER BY date DESC, symbol
        LIMIT 5
        """,
        [strategy_name],
    ).df()

    print("\n5 most recent signals:")
    if recent.empty:
        print("  None")
    else:
        for _, row in recent.iterrows():
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
            print(
                f"  {date_str} | {row['symbol']} | {row['signal_type']} | "
                f"price={row['price']:.2f} | {row['reason']}"
            )


if __name__ == "__main__":
    from strategies.registry import get_strategy

    strategy = get_strategy("sma_crossover")()

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        run_strategy(conn, strategy=strategy)
        summarize_signals(conn, strategy_name=strategy.name)
    finally:
        conn.close()

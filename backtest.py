import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import duckdb
import numpy as np
import pandas as pd

from src.strategy import ensure_signals_schema
from src.universe import get_active_universe

DEFAULT_DB_PATH: Path = Path("data/trading_data.duckdb")

# Flat depository (DP) charge per delivery SELL, before GST. Broker-dependent:
# Rs 20 is Upstox's (FY2026-27); other brokers charge roughly Rs 13-20. Being
# flat, it weighs more on small positions.
DP_CHARGE_PER_SELL: float = 20.0

def calculate_transaction_cost(trade_value: float, side: str) -> float:
    """
    Calculate approximate transaction costs for Indian equity delivery trades.
    
    Rates used (checked 2026-10 against Upstox's own brokerage calculator,
    GET /v2/charges/brokerage, for NSE equity delivery on one account):
    - Brokerage: ₹0 for delivery (that account's plan; many brokers, Upstox's
      standard tariff included, charge a per-order fee, which this omits)
    - STT: 0.1% on both buy and sell
    - Exchange Transaction Charges: 0.00307% (NSE 0.00297% + IPFT 0.0001%;
      today's rate, applied to every year of a backtest)
    - SEBI Charges: ₹10 per crore (0.0001%)
    - Stamp Duty: 0.015% on buy side only
    - DP (depository) charge: flat Rs 20 per delivery SELL (broker-dependent;
      Rs 20 is a measured Upstox FY2026-27 figure)
    - GST: 18% on (Brokerage + Exchange Charges + SEBI Charges + DP charge),
      which reproduces a real Upstox contract-note GST total
    
    NOTE: These rates change periodically and should be verified against 
    current broker and SEBI schedules.
    """
    side = side.upper()
    
    brokerage = 0.0
    stt_rate = 0.001
    exchange_rate = 0.0000307
    sebi_rate = 0.000001
    stamp_duty_rate = 0.00015
    gst_rate = 0.18
    
    stt = trade_value * stt_rate
    exchange_charges = trade_value * exchange_rate
    sebi_charges = trade_value * sebi_rate
    dp_charge = DP_CHARGE_PER_SELL if side == 'SELL' else 0.0
    
    gst = (brokerage + exchange_charges + sebi_charges + dp_charge) * gst_rate
    
    stamp_duty = (trade_value * stamp_duty_rate) if side == 'BUY' else 0.0
    
    total_cost = brokerage + stt + exchange_charges + sebi_charges + dp_charge + gst + stamp_duty
    return total_cost

def calculate_metrics(
    trades_df: pd.DataFrame, 
    equity_curve: pd.DataFrame, 
    initial_capital: float, 
    risk_free_rate: float = 0.06
) -> Dict[str, Any]:
    """
    Compute performance metrics from trades and the daily equity curve.
    """
    if trades_df.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "total_return_pct": 0.0,
            "cagr": 0.0, "max_drawdown_pct": 0.0, "sharpe_ratio": 0.0,
            "final_equity": equity_curve['equity'].iloc[-1] if not equity_curve.empty else initial_capital
        }

    # Basic trade metrics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['net_pnl'] > 0])
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    
    # Equity curve metrics
    final_equity = equity_curve['equity'].iloc[-1]
    total_return_pct = ((final_equity / initial_capital) - 1) * 100
    
    # CAGR
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days > 0:
        cagr = ((final_equity / initial_capital) ** (365.0 / days) - 1) * 100
    else:
        cagr = 0.0
        
    # Max Drawdown
    rolling_max = equity_curve['equity'].cummax()
    drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    # Sharpe Ratio (Annualized)
    # Daily returns
    daily_returns = equity_curve['equity'].pct_change().dropna()
    if not daily_returns.empty and daily_returns.std() != 0:
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = daily_returns - daily_rf
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "final_equity": final_equity
    }


def ensure_backtest_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the ``backtest_runs``, ``backtest_trades``, and ``backtest_results`` tables if missing.

    ``backtest_runs`` holds run configuration, ``backtest_trades`` holds one
    row per closed trade, and ``backtest_results`` holds one row of summary
    performance metrics per run — all keyed by ``run_id``.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id           VARCHAR NOT NULL PRIMARY KEY,
            strategy_name    VARCHAR NOT NULL,
            start_date       DATE NOT NULL,
            end_date         DATE NOT NULL,
            initial_capital  DOUBLE NOT NULL,
            position_sizing  VARCHAR NOT NULL,
            created_at       TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            trade_id      VARCHAR NOT NULL PRIMARY KEY,
            run_id        VARCHAR NOT NULL,
            symbol        VARCHAR NOT NULL,
            signal_date   DATE NOT NULL,
            entry_date    DATE NOT NULL,
            entry_price   DOUBLE NOT NULL,
            quantity      BIGINT NOT NULL,
            entry_cost    DOUBLE NOT NULL,
            exit_date     DATE NOT NULL,
            exit_price    DOUBLE NOT NULL,
            exit_cost     DOUBLE NOT NULL,
            exit_reason   VARCHAR NOT NULL,
            gross_pnl     DOUBLE NOT NULL,
            net_pnl       DOUBLE NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_results (
            run_id            VARCHAR NOT NULL PRIMARY KEY,
            total_trades      INTEGER NOT NULL,
            win_rate          DOUBLE NOT NULL,
            total_return_pct  DOUBLE NOT NULL,
            cagr              DOUBLE NOT NULL,
            max_drawdown_pct  DOUBLE NOT NULL,
            sharpe_ratio      DOUBLE NOT NULL,
            final_equity      DOUBLE NOT NULL,
            created_at        TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def _normalize_symbols(symbols: list[str] | None) -> list[str] | None:
    """Strip the ``.NS`` suffix so symbols match the storage convention."""
    if symbols is None:
        return None
    return [symbol.removesuffix(".NS") if symbol.endswith(".NS") else symbol for symbol in symbols]


def _load_signals(
    conn: duckdb.DuckDBPyConnection,
    strategy_name: str,
    start_date: str,
    end_date: str,
    symbols: list[str] | None,
) -> pd.DataFrame:
    """Load signal events for one strategy within a date range.

    Args:
        conn: Open DuckDB connection.
        strategy_name: Value in ``signals.strategy`` to filter on.
        start_date: Inclusive lower bound on ``signals.date``.
        end_date: Inclusive upper bound on ``signals.date``.
        symbols: Optional storage-symbol filter (no ``.NS`` suffix).

    Returns:
        DataFrame with ``symbol``, ``date``, ``signal_type`` columns, sorted
        by date then symbol.
    """
    ensure_signals_schema(conn)

    query = """
        SELECT symbol, date, signal_type
        FROM signals
        WHERE strategy = ?
          AND date BETWEEN ? AND ?
    """
    params: list[object] = [strategy_name, start_date, end_date]
    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)
    query += " ORDER BY date, symbol"

    signals_df = conn.execute(query, params).df()
    signals_df["date"] = pd.to_datetime(signals_df["date"]).dt.normalize()
    return signals_df


def _load_daily_prices(
    conn: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load daily open/close prices used for execution fills and mark-to-market.

    Args:
        conn: Open DuckDB connection.
        symbols: Storage symbols (no ``.NS`` suffix) to load.
        start_date: Inclusive lower bound on the candle date.
        end_date: Inclusive upper bound on the candle date.

    Returns:
        DataFrame with ``symbol``, ``date``, ``open``, ``close`` columns,
        sorted by symbol then date.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "date", "open", "close"])

    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, timestamp::DATE AS date, open, close
        FROM ohlcv_data
        WHERE timeframe = '1d'
          AND symbol IN ({placeholders})
          AND timestamp::DATE BETWEEN ? AND ?
        ORDER BY symbol, date
    """
    params: list[object] = [*symbols, start_date, end_date]
    price_df = conn.execute(query, params).df()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    return price_df


def _build_price_index(price_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index OHLC rows per symbol by date (real trading days only)."""
    return {
        symbol: group.sort_values("date").set_index("date")
        for symbol, group in price_df.groupby("symbol", sort=False)
    }


def _next_trading_day(dates: pd.Index, after: pd.Timestamp) -> pd.Timestamp | None:
    """Return the first date strictly after ``after``, or ``None`` if none exists."""
    pos = dates.searchsorted(after, side="right")
    return dates[pos] if pos < len(dates) else None


def _schedule_executions(
    signals_df: pd.DataFrame,
    price_index: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Map each signal to its lookahead-safe execution date.

    A signal dated ``T`` executes on the next available trading day *for that
    symbol* after ``T`` — never on ``T`` itself. Signals with no symbol price
    data, or no trading day after ``T`` within the loaded window, are dropped
    (they cannot be executed inside the backtest range).

    Returns:
        Executions sorted by ``exec_date``, then SELLs before BUYs, then
        ``symbol``, each a dict with
        ``symbol``, ``signal_date``, ``signal_type``, ``exec_date``.
    """
    scheduled: list[dict[str, object]] = []
    for row in signals_df.itertuples(index=False):
        symbol_prices = price_index.get(row.symbol)
        if symbol_prices is None or symbol_prices.empty:
            continue

        signal_date = pd.Timestamp(row.date).normalize()
        exec_date = _next_trading_day(symbol_prices.index, signal_date)
        if exec_date is None:
            continue

        scheduled.append(
            {
                "symbol": row.symbol,
                "signal_date": signal_date,
                "signal_type": row.signal_type,
                "exec_date": exec_date,
            }
        )

    # SELLs before BUYs on the same morning: an exit frees its slot and cash
    # for that day's entries, whatever the tickers are called.
    scheduled.sort(key=lambda item: (item["exec_date"], item["signal_type"] != "SELL", item["symbol"]))
    return scheduled


def _affordable_quantity(budget: float, fill_price: float) -> int:
    """Largest whole-share quantity whose value plus buy-side costs fits ``budget``.

    Starts from the proportional estimate at the budget's own size, then steps
    down until the real cost of the actual order fits, so a flat (non
    proportional) fee can never make the debit exceed the budget.
    """
    if budget <= 0 or fill_price <= 0:
        return 0
    cost_rate = calculate_transaction_cost(budget, "BUY") / budget
    quantity = int(budget // (fill_price * (1 + cost_rate)))
    while quantity > 0 and quantity * fill_price + calculate_transaction_cost(quantity * fill_price, "BUY") > budget:
        quantity -= 1
    return quantity


def _close_position(
    trades: list[dict[str, object]],
    symbol: str,
    position: dict[str, object],
    exit_date: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
) -> float:
    """Record a closed trade and return the net cash proceeds from the sale."""
    quantity = position["quantity"]
    trade_value = quantity * exit_price
    exit_cost = calculate_transaction_cost(trade_value, "SELL")
    gross_pnl = (exit_price - position["entry_price"]) * quantity
    net_pnl = gross_pnl - position["entry_cost"] - exit_cost

    trades.append(
        {
            "symbol": symbol,
            "signal_date": position["signal_date"],
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "quantity": quantity,
            "entry_cost": position["entry_cost"],
            "exit_date": exit_date,
            "exit_price": exit_price,
            "exit_cost": exit_cost,
            "exit_reason": exit_reason,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
        }
    )
    return trade_value - exit_cost


def _simulate(
    scheduled: list[dict[str, object]],
    price_index: dict[str, pd.DataFrame],
    close_matrix: pd.DataFrame,
    initial_capital: float,
    slippage_pct: float,
    max_concurrent_positions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Replay scheduled executions day by day, tracking cash, positions, and equity.

    Args:
        scheduled: Executions from ``_schedule_executions``.
        price_index: Per-symbol OHLC frames indexed by date (real trading
            days only), used for fills and for locating each symbol's last
            available close on forced closes.
        close_matrix: Forward-filled close prices indexed by date with one
            column per symbol, used for daily mark-to-market.
        initial_capital: Starting cash.
        slippage_pct: Slippage in percentage points applied against fills.
        max_concurrent_positions: Cap on simultaneously open positions; also
            the equal-weight sizing divisor (of equity, not of remaining cash).

    Returns:
        Tuple of ``(trades, equity_rows, skipped_signal_count)``.
    """
    slippage_frac = slippage_pct / 100.0
    trading_calendar = close_matrix.index

    executions_by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    for item in scheduled:
        executions_by_date.setdefault(item["exec_date"], []).append(item)

    cash = float(initial_capital)
    # Equity at the previous close: the sizing basis for today's fills, known
    # before the open (never today's close, which isn't known yet).
    prior_equity = float(initial_capital)
    open_positions: dict[str, dict[str, object]] = {}
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    skipped = 0

    last_day = trading_calendar[-1]

    for day in trading_calendar:
        for item in executions_by_date.get(day, []):
            symbol = item["symbol"]
            open_price = price_index[symbol].at[day, "open"]

            if item["signal_type"] == "BUY":
                if symbol in open_positions or len(open_positions) >= max_concurrent_positions:
                    skipped += 1
                    continue

                fill_price = open_price * (1 + slippage_frac)
                budget = min(prior_equity / max_concurrent_positions, cash)
                quantity = _affordable_quantity(budget, fill_price)
                if quantity < 1:
                    skipped += 1
                    continue

                trade_value = quantity * fill_price
                entry_cost = calculate_transaction_cost(trade_value, "BUY")
                total_debit = trade_value + entry_cost
                if total_debit > cash:
                    skipped += 1
                    continue

                cash -= total_debit
                open_positions[symbol] = {
                    "signal_date": item["signal_date"],
                    "entry_date": day,
                    "entry_price": fill_price,
                    "quantity": quantity,
                    "entry_cost": entry_cost,
                }

            elif item["signal_type"] == "SELL":
                position = open_positions.pop(symbol, None)
                if position is None:
                    skipped += 1
                    continue

                fill_price = open_price * (1 - slippage_frac)
                cash += _close_position(trades, symbol, position, day, fill_price, "SIGNAL")

        if day == last_day:
            for symbol, position in list(open_positions.items()):
                # Use the forward-filled close (not the raw per-symbol row) so a
                # NULL close on the final day — e.g. a candle fetched intraday
                # before the session settled — doesn't produce a NULL exit_price.
                close_price = close_matrix.at[last_day, symbol]
                cash += _close_position(
                    trades, symbol, position, last_day, close_price, "END_OF_BACKTEST"
                )
            open_positions.clear()

        positions_value = sum(
            position["quantity"] * close_matrix.at[day, symbol]
            for symbol, position in open_positions.items()
        )
        equity_rows.append(
            {
                "date": day,
                "cash": cash,
                "positions_value": positions_value,
                "equity": cash + positions_value,
            }
        )
        # A held symbol with no known close yet makes today's mark NaN; keep
        # sizing from the last finite equity rather than feeding NaN onward.
        if np.isfinite(cash + positions_value):
            prior_equity = cash + positions_value

    return trades, equity_rows, skipped


def store_backtest_results(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    position_sizing: str,
    trades_df: pd.DataFrame,
    metrics: Dict[str, Any],
) -> None:
    """Persist one backtest run's config, trades, and summary metrics.

    Writes a single row to ``backtest_runs`` (run configuration), zero or
    more rows to ``backtest_trades`` (one per row of ``trades_df``), and a
    single row to ``backtest_results`` (from ``metrics``) — all keyed by
    ``run_id``.

    Args:
        conn: Open DuckDB connection.
        run_id: Unique identifier for this backtest run.
        strategy_name: Strategy evaluated (``signals.strategy`` value).
        start_date: Inclusive backtest start (``YYYY-MM-DD``).
        end_date: Inclusive backtest end (``YYYY-MM-DD``).
        initial_capital: Starting cash.
        position_sizing: Position sizing scheme used for the run.
        trades_df: Closed trades with the ``backtest_trades`` columns
            (``symbol``, ``signal_date``, ``entry_date``, ``entry_price``,
            ``quantity``, ``entry_cost``, ``exit_date``, ``exit_price``,
            ``exit_cost``, ``exit_reason``, ``gross_pnl``, ``net_pnl``).
            May be empty.
        metrics: Summary dict from ``calculate_metrics``.
    """
    ensure_backtest_schema(conn)

    conn.execute(
        """
        INSERT INTO backtest_runs (
            run_id, strategy_name, start_date, end_date, initial_capital, position_sizing
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [run_id, strategy_name, start_date, end_date, initial_capital, position_sizing],
    )

    conn.execute(
        """
        INSERT INTO backtest_results (
            run_id, total_trades, win_rate, total_return_pct, cagr,
            max_drawdown_pct, sharpe_ratio, final_equity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            metrics["total_trades"],
            metrics["win_rate"],
            metrics["total_return_pct"],
            metrics["cagr"],
            metrics["max_drawdown_pct"],
            metrics["sharpe_ratio"],
            metrics["final_equity"],
        ],
    )

    if trades_df.empty:
        return

    staging = trades_df.copy()
    staging.insert(0, "trade_id", [str(uuid.uuid4()) for _ in range(len(staging))])
    staging.insert(1, "run_id", run_id)

    conn.register("_backtest_trades_staging", staging)
    try:
        conn.execute(
            """
            INSERT INTO backtest_trades (
                trade_id, run_id, symbol, signal_date, entry_date, entry_price,
                quantity, entry_cost, exit_date, exit_price, exit_cost,
                exit_reason, gross_pnl, net_pnl
            )
            SELECT
                trade_id, run_id, symbol, signal_date, entry_date, entry_price,
                quantity, entry_cost, exit_date, exit_price, exit_cost,
                exit_reason, gross_pnl, net_pnl
            FROM _backtest_trades_staging
            """
        )
    finally:
        conn.unregister("_backtest_trades_staging")


def run_backtest(
    conn: duckdb.DuckDBPyConnection,
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 1_000_000,
    position_sizing: str = "equal_weight",
    slippage_pct: float = 0.05,
    symbols: list[str] | None = None,
    max_concurrent_positions: int = 10,
) -> str:
    """Run an event-driven backtest for one strategy over a date range.

    A signal dated ``T`` never fills at ``T``: it fills at the next available
    trading day's OPEN price for that symbol, which keeps execution
    lookahead-free (no querying timestamps beyond what was known at ``T``).
    ``slippage_pct`` is in percentage points (``0.05`` means 0.05%) and
    always worsens the fill — buys pay above open, sells give up below open.

    Only ``position_sizing='equal_weight'`` is supported: each BUY targets
    ``equity / max_concurrent_positions``, with equity marked at the previous
    close (``initial_capital`` on the first day), capped by available cash,
    and buys as many whole shares as that budget affords including buy-side
    costs. When cash is short of the target, the entry is sized to the cash
    left, so it can be much smaller than its peers (down to one share) and
    still occupies a slot. A BUY signal is skipped (not queued, not retried) if the
    symbol already has an open position, ``max_concurrent_positions`` is
    already reached, or the allocation can't cover even one share. A SELL
    signal for a symbol with no open position is ignored.

    Any position still open at ``end_date`` is force-closed at its own last
    available close price (no slippage applied) with
    ``exit_reason='END_OF_BACKTEST'``. ``calculate_transaction_cost`` is
    charged on every entry and exit, including forced closes.

    Args:
        conn: Open DuckDB connection.
        strategy_name: Value in ``signals.strategy`` to backtest.
        start_date: Inclusive backtest start (``YYYY-MM-DD``). Signals and
            price data outside ``[start_date, end_date]`` are not loaded, so
            a signal with no execution day inside the window is skipped.
        end_date: Inclusive backtest end (``YYYY-MM-DD``).
        initial_capital: Starting cash.
        position_sizing: Only ``'equal_weight'`` is implemented.
        slippage_pct: Slippage in percentage points applied against the fill.
        symbols: Optional symbol filter (with or without ``.NS`` suffix).
            Defaults to every symbol with signals in range.
        max_concurrent_positions: Maximum simultaneously open positions; also
            the divisor for equal-weight sizing.

    Returns:
        The generated ``run_id`` (UUID4 string). Run config is stored in
        ``backtest_runs``, closed trades in ``backtest_trades``, and summary
        metrics in ``backtest_results``, all keyed by this ``run_id``.
    """
    if position_sizing != "equal_weight":
        raise ValueError(
            f"Unsupported position_sizing: {position_sizing!r}. "
            "Only 'equal_weight' is implemented."
        )

    ensure_backtest_schema(conn)

    normalized_symbols = _normalize_symbols(symbols)
    signals_df = _load_signals(conn, strategy_name, start_date, end_date, normalized_symbols)

    target_symbols = (
        normalized_symbols
        if normalized_symbols is not None
        else sorted(signals_df["symbol"].unique().tolist())
    )
    price_df = _load_daily_prices(conn, target_symbols, start_date, end_date)

    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    skipped = 0

    if not signals_df.empty and not price_df.empty:
        price_index = _build_price_index(price_df)
        close_matrix = (
            price_df.pivot(index="date", columns="symbol", values="close")
            .sort_index()
            .ffill()
        )
        scheduled = _schedule_executions(signals_df, price_index)
        trades, equity_rows, skipped = _simulate(
            scheduled=scheduled,
            price_index=price_index,
            close_matrix=close_matrix,
            initial_capital=initial_capital,
            slippage_pct=slippage_pct,
            max_concurrent_positions=max_concurrent_positions,
        )

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve = equity_curve.set_index("date")

    metrics = calculate_metrics(trades_df, equity_curve, initial_capital)

    run_id = str(uuid.uuid4())
    store_backtest_results(
        conn=conn,
        run_id=run_id,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        position_sizing=position_sizing,
        trades_df=trades_df,
        metrics=metrics,
    )

    print(f"\nBacktest run: {run_id}  ({strategy_name}, {start_date} to {end_date})")
    print(f"  Signals in range: {len(signals_df)}  |  Skipped executions: {skipped}")
    print(f"  Trades closed:    {metrics['total_trades']}  |  Win rate: {metrics['win_rate']:.1%}")
    print(
        f"  Final equity:     {metrics['final_equity']:.2f}  "
        f"({metrics['total_return_pct']:.2f}% total return)"
    )
    print(
        f"  Max drawdown:     {metrics['max_drawdown_pct']:.2f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}"
    )

    return run_id


if __name__ == "__main__":
    from strategies.registry import get_strategy

    # run_backtest() itself only needs a strategy_name string — it reads
    # already-persisted rows from `signals` and never instantiates a
    # strategy — but the registry is the source of truth for what that
    # string is for a given config, so use it here rather than hardcoding
    # "sma_crossover_20_50" and risking drift if the naming scheme changes.
    STRATEGY_NAME = get_strategy("sma_crossover")().name

    conn = duckdb.connect(str(DEFAULT_DB_PATH))
    try:
        universe_symbols = [
            ticker.removesuffix(".NS") for ticker in get_active_universe(conn)
        ]

        data_start, data_end = conn.execute(
            "SELECT MIN(timestamp)::DATE, MAX(timestamp)::DATE FROM ohlcv_data WHERE timeframe = '1d'"
        ).fetchone()

        print(f"=== Backtest: {STRATEGY_NAME} across {len(universe_symbols)} Nifty 50 symbols ===")
        print(f"Full available date range: {data_start} to {data_end}\n")

        run_id = run_backtest(
            conn,
            strategy_name=STRATEGY_NAME,
            start_date=str(data_start),
            end_date=str(data_end),
            symbols=universe_symbols,
        )

        metrics_row = conn.execute(
            """
            SELECT total_trades, win_rate, total_return_pct, cagr,
                   max_drawdown_pct, sharpe_ratio, final_equity
            FROM backtest_results
            WHERE run_id = ?
            """,
            [run_id],
        ).df()
        metrics_dict = metrics_row.iloc[0].to_dict()

        print("\n=== Metrics ===")
        print(metrics_dict)

        trades = conn.execute(
            """
            SELECT symbol, signal_date, entry_date, entry_price, quantity,
                   exit_date, exit_price, exit_reason, gross_pnl, net_pnl
            FROM backtest_trades
            WHERE run_id = ?
            """,
            [run_id],
        ).df()

        print("\n=== Top 5 trades by net PnL ===")
        print(trades.nlargest(5, "net_pnl").to_string(index=False))

        print("\n=== Bottom 5 trades by net PnL ===")
        print(trades.nsmallest(5, "net_pnl").to_string(index=False))
    finally:
        conn.close()

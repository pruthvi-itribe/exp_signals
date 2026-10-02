"""Correctness tests for backtest.py against hand-computable synthetic data.

Every test seeds an in-memory DuckDB connection (or, for test 7, real
already-fetched OHLCV rows copied into an in-memory connection) with small,
fully-specified data so the expected result can be worked out by hand and
hardcoded as an assertion -- not eyeballed for plausibility. Each test's
docstring says what specific bug it would catch if it failed.

Note: `ohlcv_data` is the actual table name in this project (filtered by
`timeframe = '1d'`) -- there is no separate `ohlcv_daily` table.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

import backtest
from backtest import calculate_metrics, calculate_transaction_cost, run_backtest
from tests.helpers import REAL_DB_PATH, insert_ohlcv, insert_signal, make_conn, zero_cost

# Local aliases so the rest of this file (written before helpers.py existed)
# doesn't need touching beyond the import -- new test files should call the
# tests.helpers names directly instead of aliasing them like this.
_make_conn = make_conn
_insert_ohlcv = insert_ohlcv
_insert_signal = insert_signal
_zero_cost = zero_cost


def test_single_trade_no_costs(monkeypatch):
    """One BUY->SELL round trip with slippage and transaction costs zeroed out.

    Would catch: a wrong T+1 fill price, a wrong quantity/allocation
    calculation, or gross_pnl != net_pnl when costs are genuinely zero (e.g.
    a stray cost being added even though calculate_transaction_cost returns 0).
    """
    _zero_cost(monkeypatch)

    conn = _make_conn()
    symbol = "AAACO"
    strategy = "test_strategy"

    prices = [
        ("2024-01-01", 100, 101),
        ("2024-01-02", 102, 103),  # BUY signal fires here
        ("2024-01-03", 105, 104),  # T+1: entry executes at this OPEN (105)
        ("2024-01-04", 106, 108),
        ("2024-01-05", 110, 111),  # SELL signal fires here
        ("2024-01-06", 112, 113),  # T+1: exit executes at this OPEN (112)
        ("2024-01-07", 114, 115),
        ("2024-01-08", 116, 117),
        ("2024-01-09", 118, 119),
        ("2024-01-10", 120, 121),
    ]
    _insert_ohlcv(conn, symbol, prices)
    _insert_signal(conn, symbol, "2024-01-02", strategy, "BUY")
    _insert_signal(conn, symbol, "2024-01-05", strategy, "SELL")

    run_id = run_backtest(
        conn,
        strategy_name=strategy,
        start_date="2024-01-01",
        end_date="2024-01-10",
        initial_capital=100_000,
        slippage_pct=0.0,
        symbols=[symbol],
        max_concurrent_positions=1,
    )

    trade = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", [run_id]).df().iloc[0]

    # Hand-computed: allocation = 100000 / 1 = 100000; fill_price = open = 105
    # (no slippage) -> quantity = int(100000 // 105) = 952.
    expected_quantity = 952
    expected_entry_price = 105.0
    expected_exit_price = 112.0
    expected_gross_pnl = (expected_exit_price - expected_entry_price) * expected_quantity  # 6664.0

    assert trade["symbol"] == symbol
    assert pd.Timestamp(trade["signal_date"]) == pd.Timestamp("2024-01-02")
    assert pd.Timestamp(trade["entry_date"]) == pd.Timestamp("2024-01-03")
    assert trade["entry_price"] == pytest.approx(expected_entry_price)
    assert trade["quantity"] == expected_quantity
    assert trade["entry_cost"] == pytest.approx(0.0)
    assert pd.Timestamp(trade["exit_date"]) == pd.Timestamp("2024-01-06")
    assert trade["exit_price"] == pytest.approx(expected_exit_price)
    assert trade["exit_cost"] == pytest.approx(0.0)
    assert trade["exit_reason"] == "SIGNAL"
    assert trade["gross_pnl"] == pytest.approx(expected_gross_pnl)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"])  # costs are zero
    assert trade["net_pnl"] == pytest.approx(expected_gross_pnl)

    conn.close()


def test_t_plus_1_execution_timing(monkeypatch):
    """A signal dated T fills at T+1's OPEN, never at T's own close, and T+1
    means the symbol's next *actual* row in the data, not calendar date+1.

    Part A: T's close (100) and T+1's open (150) are deliberately far apart
    -- would catch a lookahead bug that fills at T's close (or T's own open).
    Part B: the signal fires on the last row before a multi-day gap in the
    loaded data -- would catch execution scheduling that assumes date+1
    instead of searching the symbol's actual next available trading day.

    Costs are zeroed out: this test is about *timing*, not cost interaction,
    and real costs on top of a snugly-sized fill can tip total_debit over
    available cash and skip the BUY entirely, which would be a false failure
    unrelated to what this test checks.
    """
    _zero_cost(monkeypatch)
    strategy = "test_strategy"

    # --- Part A ---
    conn = _make_conn()
    symbol = "TIMECO"
    prices = [
        ("2024-01-01", 90, 100),   # BUY signal here; close = 100
        ("2024-01-02", 150, 160),  # T+1 open = 150 -- this must be the fill
        ("2024-01-03", 155, 158),  # end_date; no SELL -> force-closed here
    ]
    _insert_ohlcv(conn, symbol, prices)
    _insert_signal(conn, symbol, "2024-01-01", strategy, "BUY")

    run_id = run_backtest(
        conn,
        strategy_name=strategy,
        start_date="2024-01-01",
        end_date="2024-01-03",
        initial_capital=100_000,
        slippage_pct=0.0,
        symbols=[symbol],
        max_concurrent_positions=1,
    )
    trade = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", [run_id]).df().iloc[0]
    assert trade["entry_price"] == pytest.approx(150.0)
    assert trade["entry_price"] != pytest.approx(100.0)
    assert pd.Timestamp(trade["entry_date"]) == pd.Timestamp("2024-01-02")
    conn.close()

    # --- Part B: signal on the last day before a data gap (long weekend) ---
    conn = _make_conn()
    symbol = "GAPCO"
    prices = [
        ("2024-01-05", 200, 205),  # Friday; BUY signal here
        # 2024-01-06/07 (weekend) and 2024-01-08 (holiday) have no rows at all.
        ("2024-01-09", 210, 212),  # next ACTUAL trading day in the data
        ("2024-01-10", 213, 214),
    ]
    _insert_ohlcv(conn, symbol, prices)
    _insert_signal(conn, symbol, "2024-01-05", strategy, "BUY")

    run_id = run_backtest(
        conn,
        strategy_name=strategy,
        start_date="2024-01-05",
        end_date="2024-01-10",
        initial_capital=100_000,
        slippage_pct=0.0,
        symbols=[symbol],
        max_concurrent_positions=1,
    )
    trade = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", [run_id]).df().iloc[0]
    assert pd.Timestamp(trade["entry_date"]) == pd.Timestamp("2024-01-09")
    assert trade["entry_price"] == pytest.approx(210.0)
    conn.close()


def test_transaction_cost_calculation():
    """calculate_transaction_cost() matches its own documented STT/exchange/
    SEBI/GST/stamp-duty formula exactly, for both BUY and SELL.

    Would catch: a wrong rate constant, GST computed on the wrong base,
    stamp duty leaking into SELL trades (or missing from BUY trades -- it is
    documented as buy-side only), or a case-sensitive side check.
    """
    # Hand-computed expected values (trade_value * rate, summed per the
    # docstring's formula):
    #   BUY  100000: stt=100 + exch=3.45 + sebi=0.1 + gst=0.621 + stamp=15   = 119.171
    #   SELL 100000: stt=100 + exch=3.45 + sebi=0.1 + gst=0.621 + stamp=0    = 104.171
    #   BUY   50000: stt=50  + exch=1.725+ sebi=0.05+ gst=0.3105+ stamp=7.5  =  59.5855
    #   SELL  50000: stt=50  + exch=1.725+ sebi=0.05+ gst=0.3105+ stamp=0    =  52.0855
    assert calculate_transaction_cost(100_000.0, "BUY") == pytest.approx(119.171, rel=1e-9)
    assert calculate_transaction_cost(100_000.0, "SELL") == pytest.approx(104.171, rel=1e-9)
    assert calculate_transaction_cost(50_000.0, "BUY") == pytest.approx(59.5855, rel=1e-9)
    assert calculate_transaction_cost(50_000.0, "SELL") == pytest.approx(52.0855, rel=1e-9)

    # Independent reference implementation of the same documented formula,
    # cross-checked against a wider set of trade values.
    def expected_cost(trade_value: float, side: str) -> float:
        stt = trade_value * 0.001
        exchange_charges = trade_value * 0.0000345
        sebi_charges = trade_value * 0.000001
        gst = exchange_charges * 0.18  # brokerage is 0
        stamp_duty = trade_value * 0.00015 if side == "BUY" else 0.0
        return stt + exchange_charges + sebi_charges + gst + stamp_duty

    for trade_value in (1_000.0, 25_000.0, 250_000.0, 1_000_000.0):
        for side in ("BUY", "SELL"):
            assert calculate_transaction_cost(trade_value, side) == pytest.approx(
                expected_cost(trade_value, side), rel=1e-9
            )

    # side must not be case-sensitive.
    assert calculate_transaction_cost(100_000.0, "buy") == pytest.approx(
        calculate_transaction_cost(100_000.0, "BUY")
    )


def test_multiple_concurrent_positions_capital_allocation(monkeypatch):
    """3 symbols BUY on the same day at the same price, max_concurrent_positions=3.

    Equal-weight sizing recomputes `cash / max_concurrent_positions` from
    *current* cash for every fill (per run_backtest's own docstring), not a
    fixed pre-split budget -- so within one day, the alphabetically-first
    symbol (per _schedule_executions's sort) gets a full 1/3 of capital, the
    next gets 1/3 of what's left, etc. This pins down that exact sequential
    result by hand.

    Would catch: a double-spend bug (each position computing its share of
    the *original* capital independently, letting total allocation exceed
    initial_capital) or a sizing divisor that ignores max_concurrent_positions.
    """
    _zero_cost(monkeypatch)

    conn = _make_conn()
    strategy = "test_strategy"
    symbols = ["AAA", "BBB", "CCC"]
    for symbol in symbols:
        _insert_ohlcv(
            conn,
            symbol,
            [
                ("2024-01-01", 90, 95),
                ("2024-01-02", 100, 102),  # all 3 BUY signals execute here, open=100
                ("2024-01-03", 101, 103),
            ],
        )
        _insert_signal(conn, symbol, "2024-01-01", strategy, "BUY")

    run_id = run_backtest(
        conn,
        strategy_name=strategy,
        start_date="2024-01-01",
        end_date="2024-01-03",
        initial_capital=300_000,
        slippage_pct=0.0,
        symbols=symbols,
        max_concurrent_positions=3,
    )

    trades = conn.execute(
        "SELECT symbol, quantity, entry_price FROM backtest_trades WHERE run_id = ? ORDER BY symbol",
        [run_id],
    ).df()

    # Hand-computed sequential allocation (fill_price = 100 for all three):
    #   AAA: alloc = 300000/3 = 100000.00 -> qty = int(100000.00/100) = 1000, value=100000, cash -> 200000
    #   BBB: alloc = 200000/3 =  66666.67 -> qty = int( 66666.67/100) =  666, value= 66600, cash -> 133400
    #   CCC: alloc = 133400/3 =  44466.67 -> qty = int( 44466.67/100) =  444, value= 44400, cash ->  89000
    expected_quantity = {"AAA": 1000, "BBB": 666, "CCC": 444}
    for _, row in trades.iterrows():
        assert row["quantity"] == expected_quantity[row["symbol"]]
        assert row["entry_price"] == pytest.approx(100.0)

    total_allocated = float((trades["quantity"] * trades["entry_price"]).sum())
    assert total_allocated == pytest.approx(100_000 + 66_600 + 44_400)
    # The double-spend guard: total allocated across all positions opened on
    # the same day must never exceed initial_capital.
    assert total_allocated <= 300_000

    conn.close()


def test_force_close_at_end_of_backtest(monkeypatch):
    """A BUY with no matching SELL before end_date is force-closed at end_date.

    Would catch: a position silently left open (never recorded in
    backtest_trades, never contributing its PnL to final equity), the wrong
    exit_reason, or a force-close price basis other than the last available
    close (e.g. using that day's open, or a stale/NULL close).
    """
    _zero_cost(monkeypatch)

    conn = _make_conn()
    symbol = "HOLDCO"
    strategy = "test_strategy"
    prices = [
        ("2024-01-01", 100, 102),  # BUY signal here
        ("2024-01-02", 103, 105),  # T+1: entry executes at open = 103
        ("2024-01-03", 106, 108),
        ("2024-01-04", 109, 110),  # end_date; last available close = 110
    ]
    _insert_ohlcv(conn, symbol, prices)
    _insert_signal(conn, symbol, "2024-01-01", strategy, "BUY")

    run_id = run_backtest(
        conn,
        strategy_name=strategy,
        start_date="2024-01-01",
        end_date="2024-01-04",
        initial_capital=100_000,
        slippage_pct=0.0,
        symbols=[symbol],
        max_concurrent_positions=1,
    )

    trade = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", [run_id]).df().iloc[0]

    expected_quantity = 970  # int(100000 // 103)
    expected_gross = (110.0 - 103.0) * expected_quantity

    assert trade["quantity"] == expected_quantity
    assert trade["entry_price"] == pytest.approx(103.0)
    assert pd.Timestamp(trade["exit_date"]) == pd.Timestamp("2024-01-04")
    assert trade["exit_price"] == pytest.approx(110.0)  # last available CLOSE, not open
    assert trade["exit_reason"] == "END_OF_BACKTEST"
    assert trade["gross_pnl"] == pytest.approx(expected_gross)
    assert trade["net_pnl"] == pytest.approx(expected_gross)

    conn.close()


def test_metrics_calculation_known_values():
    """calculate_metrics() on a hand-picked equity curve matches an
    independently-computed max drawdown and Sharpe ratio.

    Would catch: a max-drawdown formula measuring from a fixed start instead
    of the running peak, an off-by-one in the Sharpe annualization factor
    (sqrt(252)), or a sample/population std mismatch against pandas' default
    ddof=1.
    """
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    equity_curve = pd.DataFrame({"equity": [100.0, 150.0, 90.0, 120.0]}, index=dates)

    # trades_df only needs to be non-empty: calculate_metrics short-circuits
    # to all-zero metrics when it's empty, which would skip the drawdown/
    # Sharpe code path entirely.
    trades_df = pd.DataFrame([{"net_pnl": 10.0}, {"net_pnl": -5.0}])

    metrics = calculate_metrics(trades_df, equity_curve, initial_capital=100.0)

    # Hand-computed max drawdown: peak 150 -> trough 90 => (150-90)/150 = 40%.
    assert metrics["max_drawdown_pct"] == pytest.approx(40.0, abs=1e-4)

    # Hand-computed total return: (120/100 - 1) * 100 = 20%.
    assert metrics["total_return_pct"] == pytest.approx(20.0, abs=1e-4)

    # Independently-computed Sharpe using the documented formula: daily
    # returns [0.5, -0.4, 1/3], annualized daily risk-free rate from the 6%
    # default, mean/std (ddof=1, matching pandas' Series.std() default) of
    # excess returns * sqrt(252).
    daily_returns = np.array([0.5, -0.4, 1.0 / 3.0])
    daily_rf = (1 + 0.06) ** (1 / 252) - 1
    excess = daily_returns - daily_rf
    expected_sharpe = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, abs=1e-4)


def test_buy_and_hold_benchmark_matches_manual_calculation(monkeypatch):
    """The full run_backtest() pipeline, on real already-fetched OHLCV data,
    matches an independent "always long from day 1" ground-truth calculation.

    NOTE ON adj_close: backtest.py's execution reads *raw* open/close from
    ohlcv_data -- it never reads adj_close (see _load_daily_prices's query).
    In this project's data, adj_close diverges from close on essentially any
    real symbol/date range with a dividend in it (verified: RELIANCE's
    adj_close/close ratio shifts partway through 2023, in the exact window
    used below), so "first adj_close vs last adj_close" would be a different
    quantity than what run_backtest() actually computes -- comparing against
    it would fail for a reason unrelated to any backtest bug. The ground
    truth here instead mirrors the engine's own documented mechanics exactly
    (T+1 OPEN entry, end-of-window CLOSE exit, whole-share floor quantity),
    computed directly from the OHLCV rows independently of backtest.py's own
    functions.

    Would catch: any pipeline-level bug a narrower unit test could miss --
    e.g. broken wiring between _schedule_executions/_simulate/calculate_metrics,
    or equity-curve/cash bookkeeping errors that only show up over a full run.
    """
    if not REAL_DB_PATH.exists():
        pytest.skip(f"real project database not found at {REAL_DB_PATH}")

    _zero_cost(monkeypatch)

    symbol = "RELIANCE"
    start_date, end_date = "2023-01-01", "2023-12-31"

    real_conn = duckdb.connect(str(REAL_DB_PATH), read_only=True)
    try:
        price_df = real_conn.execute(
            """
            SELECT timestamp::DATE AS date, open, close
            FROM ohlcv_data
            WHERE timeframe = '1d' AND symbol = ? AND timestamp::DATE BETWEEN ? AND ?
            ORDER BY date
            """,
            [symbol, start_date, end_date],
        ).df()
    finally:
        real_conn.close()

    if len(price_df) < 2:
        pytest.skip(f"not enough real {symbol} data in {start_date}..{end_date} to run this test")

    # Ground truth mirroring run_backtest()'s documented mechanics: a BUY
    # signal on the first available date fills at the NEXT row's open; with
    # no SELL, the position is force-closed at the LAST available date's close.
    first_date = price_df["date"].iloc[0]
    entry_price = float(price_df["open"].iloc[1])
    exit_price = float(price_df["close"].iloc[-1])
    initial_capital = 10_000_000.0
    max_concurrent_positions = 1
    expected_quantity = int((initial_capital / max_concurrent_positions) // entry_price)
    expected_total_return_pct = (
        expected_quantity * (exit_price - entry_price) / initial_capital
    ) * 100

    conn = _make_conn()
    rows = [
        (d.strftime("%Y-%m-%d"), float(o), float(c))
        for d, o, c in price_df.itertuples(index=False, name=None)
    ]
    _insert_ohlcv(conn, symbol, rows)
    _insert_signal(conn, symbol, first_date.strftime("%Y-%m-%d"), "test_buy_and_hold", "BUY")

    run_id = run_backtest(
        conn,
        strategy_name="test_buy_and_hold",
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        slippage_pct=0.0,
        symbols=[symbol],
        max_concurrent_positions=max_concurrent_positions,
    )

    trade = conn.execute("SELECT * FROM backtest_trades WHERE run_id = ?", [run_id]).df().iloc[0]
    assert trade["quantity"] == expected_quantity
    assert trade["entry_price"] == pytest.approx(entry_price)
    assert trade["exit_price"] == pytest.approx(exit_price)
    assert trade["exit_reason"] == "END_OF_BACKTEST"

    result = conn.execute(
        "SELECT total_return_pct FROM backtest_results WHERE run_id = ?", [run_id]
    ).df().iloc[0]
    assert result["total_return_pct"] == pytest.approx(expected_total_return_pct, rel=1e-6)

    conn.close()

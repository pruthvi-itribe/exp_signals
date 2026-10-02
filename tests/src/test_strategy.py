"""Tests for src/strategy.py: strategy input loading, signal storage/upsert,
orchestration (run_strategy), and reporting (summarize_signals).

Would catch: a broken upsert (duplicate rows instead of in-place update), a
strategy-input join that includes ohlcv rows with no matching indicator row
(should be excluded, not NULL-padded), or an orchestration path that crashes
on empty input instead of degrading gracefully.
"""

import pandas as pd
import pytest

import src.strategy as strategy_mod
from src.indicators import ensure_indicators_schema
from src.strategy import (
    _prepare_signal_staging,
    _resolve_symbols,
    load_strategy_input,
    run_strategy,
    store_signals,
    summarize_signals,
)
from strategies.base import SIGNAL_OUTPUT_COLUMNS, Strategy
from tests.helpers import insert_ohlcv, insert_signal, make_conn


def _insert_indicator_row(conn, symbol, date_str, **overrides):
    row = {
        "daily_return": 0.0,
        "sma_20": 0.0,
        "sma_50": 0.0,
        "ema_12": 0.0,
        "ema_26": 0.0,
        "rsi_14": 50.0,
        "volatility_20": 0.0,
    }
    row.update(overrides)
    conn.execute(
        """
        INSERT INTO indicators_daily
            (symbol, date, daily_return, sma_20, sma_50, ema_12, ema_26, rsi_14, volatility_20)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            symbol, date_str, row["daily_return"], row["sma_20"], row["sma_50"],
            row["ema_12"], row["ema_26"], row["rsi_14"], row["volatility_20"],
        ],
    )


class _FixedSignalStrategy(Strategy):
    """Minimal Strategy stub that returns a pre-built signals DataFrame,
    used to test run_strategy's orchestration without depending on any real
    strategy's indicator logic."""

    base_name = "fixed_test_strategy"
    required_columns = ("symbol", "date", "adj_close")

    def __init__(self, signals_df: pd.DataFrame | None = None, **config_kwargs):
        super().__init__(**config_kwargs)
        self._signals_df = (
            signals_df if signals_df is not None else pd.DataFrame(columns=SIGNAL_OUTPUT_COLUMNS)
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._signals_df


# --- ensure_signals_schema ---


def test_ensure_signals_schema_is_idempotent():
    conn = make_conn()
    strategy_mod.ensure_signals_schema(conn)  # already created by make_conn(); calling again must not error
    conn.close()


# --- _resolve_symbols ---


def test_resolve_symbols_strips_ns_suffix_when_given_explicitly():
    conn = make_conn()
    assert _resolve_symbols(conn, ["RELIANCE.NS", "TCS"]) == ["RELIANCE", "TCS"]
    conn.close()


def test_resolve_symbols_none_defaults_to_active_universe(monkeypatch):
    conn = make_conn()
    monkeypatch.setattr(strategy_mod, "get_active_universe", lambda conn: ["INFY.NS", "WIPRO.NS"])
    assert _resolve_symbols(conn, None) == ["INFY", "WIPRO"]
    conn.close()


def test_resolve_symbols_empty_list_does_not_touch_active_universe(monkeypatch):
    """symbols=[] is explicitly "no symbols", distinct from symbols=None
    ("use the active universe") -- would catch the two being conflated."""
    conn = make_conn()

    def _boom(conn):
        raise AssertionError("get_active_universe must not be called when symbols=[] is explicit")

    monkeypatch.setattr(strategy_mod, "get_active_universe", _boom)
    assert _resolve_symbols(conn, []) == []
    conn.close()


# --- load_strategy_input ---


def test_load_strategy_input_joins_ohlcv_and_indicators():
    conn = make_conn()
    ensure_indicators_schema(conn)
    insert_ohlcv(conn, "AAA", [("2024-01-01", 100, 102), ("2024-01-02", 103, 104)])
    _insert_indicator_row(conn, "AAA", "2024-01-01", sma_20=99.5)
    # Deliberately no indicator row for 2024-01-02: the INNER JOIN must
    # exclude that date rather than including it with NULL indicator columns.
    _insert_indicator_row(conn, "AAA", "2024-01-03", sma_20=1234.0)  # no matching ohlcv row either

    df = load_strategy_input(conn, symbols=["AAA"])

    assert len(df) == 1
    assert df.iloc[0]["date"] == pd.Timestamp("2024-01-01")
    assert df.iloc[0]["sma_20"] == pytest.approx(99.5)
    conn.close()


def test_load_strategy_input_attaches_market_regime_columns():
    """Would catch: load_strategy_input not wiring in
    src.market_regime.attach_market_regime at all -- a strategy relying on
    the Nifty regime filter would then always see it as absent (inert) even
    when the index HAS been fetched, silently disabling the filter."""
    conn = make_conn()
    ensure_indicators_schema(conn)
    insert_ohlcv(conn, "AAA", [("2024-01-01", 100, 102)])
    _insert_indicator_row(conn, "AAA", "2024-01-01", sma_20=99.5)

    # No ^NSEI data seeded at all -- must fail open (inert), not crash or
    # silently block everything.
    df = load_strategy_input(conn, symbols=["AAA"])
    assert df.iloc[0]["nifty_regime_bullish"] == True  # noqa: E712
    assert df.iloc[0]["nifty_regime_breakdown"] == False  # noqa: E712
    conn.close()


def test_load_strategy_input_empty_symbols_list_returns_empty_df():
    conn = make_conn()
    ensure_indicators_schema(conn)
    df = load_strategy_input(conn, symbols=[])
    assert df.empty
    conn.close()


def test_load_strategy_input_no_data_for_symbol_returns_empty_df():
    conn = make_conn()
    ensure_indicators_schema(conn)
    df = load_strategy_input(conn, symbols=["NOPE"])
    assert df.empty
    conn.close()


# --- _prepare_signal_staging ---


def test_prepare_signal_staging_missing_columns_raises():
    df = pd.DataFrame([{"symbol": "AAA", "date": "2024-01-01", "strategy": "s"}])
    with pytest.raises(ValueError, match="signal_type"):
        _prepare_signal_staging(df)


def test_prepare_signal_staging_normalizes_date_and_sets_generated_at():
    df = pd.DataFrame(
        [{"symbol": "AAA", "date": "2024-01-01 15:30:00", "strategy": "s",
          "signal_type": "BUY", "price": 1.0, "reason": "r"}]
    )
    staging = _prepare_signal_staging(df)
    assert staging.iloc[0]["date"] == pd.Timestamp("2024-01-01")  # normalized to midnight
    # generated_at is UTC-based (datetime.now(timezone.utc).replace(tzinfo=None)),
    # so it must be compared against UTC now, not local wall-clock time.
    utc_now_naive = pd.Timestamp.now("UTC").tz_localize(None)
    assert (utc_now_naive - staging.iloc[0]["generated_at"]) < pd.Timedelta(minutes=5)


# --- store_signals ---


def test_store_signals_empty_input_returns_zero_summary():
    conn = make_conn()
    summary = store_signals(conn, pd.DataFrame(columns=SIGNAL_OUTPUT_COLUMNS))
    assert summary == {"signals_inserted": 0, "buy_count": 0, "sell_count": 0}
    conn.close()


def test_store_signals_upsert_replaces_not_duplicates():
    conn = make_conn()
    df1 = pd.DataFrame(
        [{"symbol": "AAA", "date": "2024-01-01", "strategy": "s",
          "signal_type": "BUY", "price": 10.0, "reason": "r1"}]
    )
    summary1 = store_signals(conn, df1)
    assert summary1 == {"signals_inserted": 1, "buy_count": 1, "sell_count": 0}

    df2 = pd.DataFrame(
        [{"symbol": "AAA", "date": "2024-01-01", "strategy": "s",
          "signal_type": "SELL", "price": 20.0, "reason": "r2"}]
    )
    summary2 = store_signals(conn, df2)
    # Same (symbol, date, strategy) key -> this is an update, not a fresh insert.
    assert summary2 == {"signals_inserted": 0, "buy_count": 0, "sell_count": 1}

    rows = conn.execute(
        "SELECT signal_type, price, reason FROM signals WHERE symbol='AAA' AND date='2024-01-01' AND strategy='s'"
    ).df()
    assert len(rows) == 1  # exactly one row survives -- replaced, not duplicated
    assert rows.iloc[0]["signal_type"] == "SELL"
    assert rows.iloc[0]["price"] == pytest.approx(20.0)
    assert rows.iloc[0]["reason"] == "r2"
    conn.close()


# --- run_strategy ---


def test_run_strategy_empty_input_returns_zero_and_prints_message(capsys):
    conn = make_conn()
    ensure_indicators_schema(conn)
    strategy = _FixedSignalStrategy()
    summary = run_strategy(conn, strategy, symbols=["NOPE"])
    assert summary == {"signals_inserted": 0, "buy_count": 0, "sell_count": 0}
    assert "No strategy input rows found" in capsys.readouterr().out
    conn.close()


def test_run_strategy_full_flow_persists_signals(capsys):
    conn = make_conn()
    ensure_indicators_schema(conn)
    insert_ohlcv(conn, "AAA", [("2024-01-01", 100, 102)])
    _insert_indicator_row(conn, "AAA", "2024-01-01")

    signals_df = pd.DataFrame(
        [{"symbol": "AAA", "date": "2024-01-01", "strategy": "fixed_test_strategy",
          "signal_type": "BUY", "price": 100.0, "reason": "test"}]
    )
    strategy = _FixedSignalStrategy(signals_df=signals_df)
    summary = run_strategy(conn, strategy, symbols=["AAA"])

    assert summary == {"signals_inserted": 1, "buy_count": 1, "sell_count": 0}
    stored = conn.execute("SELECT symbol, signal_type, price, reason FROM signals").df()
    assert len(stored) == 1
    assert stored.iloc[0]["signal_type"] == "BUY"
    assert "Signals emitted: 1" in capsys.readouterr().out
    conn.close()


# --- summarize_signals ---


def test_summarize_signals_no_signals_prints_message(capsys):
    conn = make_conn()
    summarize_signals(conn, "nope")
    assert "No signals stored for this strategy." in capsys.readouterr().out
    conn.close()


def test_summarize_signals_totals_and_per_symbol_breakdown(capsys):
    conn = make_conn()
    insert_signal(conn, "AAA", "2024-01-01", "s", "BUY")
    insert_signal(conn, "AAA", "2024-01-02", "s", "SELL")
    insert_signal(conn, "BBB", "2024-01-01", "s", "BUY")

    summarize_signals(conn, "s")
    out = capsys.readouterr().out

    assert "Total BUY:  2" in out
    assert "Total SELL: 1" in out
    assert "AAA: BUY=1, SELL=1" in out
    assert "BBB: BUY=1, SELL=0" in out
    conn.close()

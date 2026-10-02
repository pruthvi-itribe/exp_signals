"""Correctness tests for research/screen.py.

verdict_for/_parse_params/_comparison_table are pure functions tested with
hand-picked boundary values. Data-loading helpers and screen_signal are
tested against tests.helpers.make_conn() with small synthetic panels -- never
the real project DB or a real network call.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from research.forward_returns import compute_and_store_forward_returns, ensure_forward_returns_schema
from research.screen import (
    MIN_DAYS_FOR_VERDICT,
    STRONG_IC_THRESHOLD,
    T_STAT_SIGNIFICANCE,
    WEAK_IC_THRESHOLD,
    _build_parser,
    _comparison_table,
    _load_forward_returns,
    _load_ohlcv_history,
    _parse_params,
    _resolve_date_range,
    _resolve_universe_symbols,
    screen_signal,
    verdict_for,
)
from tests.helpers import insert_ohlcv, make_conn


# ---------------------------------------------------------------------------
# verdict_for -- exact threshold boundaries
# ---------------------------------------------------------------------------


def test_verdict_for_insufficient_data_boundary():
    """n_days below MIN_DAYS_FOR_VERDICT (20) is always insufficient, even
    with an otherwise-excellent IC/t-stat. n_days == 20 is NOT insufficient
    (the check is a strict '<'). Would catch an off-by-one on the sample-size gate."""
    assert verdict_for(0.06, 3.0, MIN_DAYS_FOR_VERDICT - 1) == "Insufficient data to draw a conclusion."
    assert verdict_for(0.06, 3.0, MIN_DAYS_FOR_VERDICT) != "Insufficient data to draw a conclusion."


def test_verdict_for_nan_inputs_are_insufficient():
    """A NaN mean_ic or t_stat (e.g. a zero-variance IC series) must not slip
    past the insufficient-data gate into a false 'no edge'/'edge' verdict."""
    assert verdict_for(float("nan"), 3.0, 25) == "Insufficient data to draw a conclusion."
    assert verdict_for(0.06, float("nan"), 25) == "Insufficient data to draw a conclusion."


def test_verdict_for_significance_and_magnitude_boundaries():
    """Exact boundary values for T_STAT_SIGNIFICANCE (2.0) and
    WEAK_IC_THRESHOLD (0.02)/STRONG_IC_THRESHOLD (0.05) -- each is a strict
    '<' so the threshold value itself passes, one tick below does not.
    Would catch an inclusive/exclusive boundary flip on any of the three cuts."""
    # exactly at both gates -> passes them, magnitude below STRONG -> "weak"
    assert verdict_for(WEAK_IC_THRESHOLD, T_STAT_SIGNIFICANCE, 20) == (
        "Weak but potentially real signal — worth a closer look before building a strategy."
    )
    # t-stat just below significance -> no edge, even with a fine magnitude
    assert verdict_for(WEAK_IC_THRESHOLD, T_STAT_SIGNIFICANCE - 0.01, 20) == "No meaningful edge detected."
    # magnitude just below weak threshold -> no edge, even with a fine t-stat
    assert verdict_for(WEAK_IC_THRESHOLD - 0.0001, T_STAT_SIGNIFICANCE, 20) == "No meaningful edge detected."
    # exactly at strong threshold -> NOT "weak" (strict '<'), promoted to full edge
    assert verdict_for(STRONG_IC_THRESHOLD, T_STAT_SIGNIFICANCE, 20) == (
        "Signal shows edge, worth building a strategy around."
    )
    # negative mean_ic uses abs() -- same thresholds apply by magnitude
    assert verdict_for(-0.06, -3.0, 25) == "Signal shows edge, worth building a strategy around."


# ---------------------------------------------------------------------------
# _parse_params
# ---------------------------------------------------------------------------


def test_parse_params_empty_and_none():
    """Would catch None vs '' being handled inconsistently."""
    assert _parse_params(None) == {}
    assert _parse_params("") == {}


def test_parse_params_type_coercion_and_whitespace():
    """Values coerce int -> float -> raw string, in that order, with keys/values stripped."""
    assert _parse_params("window=20,num_std=2.0") == {"window": 20, "num_std": 2.0}
    assert _parse_params(" window = 20 , mode = aggressive ") == {"window": 20, "mode": "aggressive"}
    assert _parse_params("window=-5,thresh=-2.5") == {"window": -5, "thresh": -2.5}


def test_parse_params_empty_value_falls_back_to_string():
    """A key with no value (e.g. a trailing comma or malformed pair) must
    not raise -- both int() and float() on '' raise ValueError, so it falls
    through to the raw (empty) string, not crash the CLI."""
    assert _parse_params("window=,x=5") == {"window": "", "x": 5}


# ---------------------------------------------------------------------------
# _resolve_universe_symbols
# ---------------------------------------------------------------------------


def test_resolve_universe_symbols_nifty50_uses_active_universe(monkeypatch):
    """'nifty50' (any case) must resolve via get_active_universe, not be
    treated as a literal comma-separated symbol list."""
    import research.screen as screen_module

    monkeypatch.setattr(screen_module, "get_active_universe", lambda conn: ["AAA.NS", "BBB.NS"])
    conn = duckdb.connect(":memory:")
    assert _resolve_universe_symbols(conn, "NIFTY50") == ["AAA", "BBB"]
    conn.close()


def test_resolve_universe_symbols_custom_list_strips_ns_and_whitespace():
    """Would catch a stray space or un-stripped .NS suffix leaking into a
    downstream SQL IN (...) filter that then matches nothing."""
    conn = duckdb.connect(":memory:")
    assert _resolve_universe_symbols(conn, "RELIANCE.NS, TCS , INFY.NS") == ["RELIANCE", "TCS", "INFY"]
    conn.close()


# ---------------------------------------------------------------------------
# _resolve_date_range
# ---------------------------------------------------------------------------


def test_resolve_date_range_explicit_bounds_pass_through():
    """Would catch explicit --start/--end being overridden by the data-derived default."""
    conn = make_conn()
    result = _resolve_date_range(conn, ["AAA"], "2020-01-01", "2020-12-31")
    assert result == ("2020-01-01", "2020-12-31")
    conn.close()


def test_resolve_date_range_defaults_to_available_data_range():
    """No --start/--end -- defaults to MIN/MAX of the loaded OHLCV data."""
    conn = make_conn()
    insert_ohlcv(conn, "AAA", [("2024-01-01", 100.0, 101.0), ("2024-03-15", 110.0, 111.0)])
    start, end = _resolve_date_range(conn, ["AAA"], None, None)
    assert start == "2024-01-01"
    assert end == "2024-03-15"
    conn.close()


def test_resolve_date_range_no_data_raises():
    """No OHLCV rows at all for the universe -- would catch a silent
    (None, None) range being passed downstream instead of a clear error."""
    conn = make_conn()
    with pytest.raises(RuntimeError, match="No OHLCV data"):
        _resolve_date_range(conn, ["NOPE"], None, None)
    conn.close()


# ---------------------------------------------------------------------------
# _load_ohlcv_history / _load_forward_returns
# ---------------------------------------------------------------------------


def test_load_ohlcv_history_respects_end_date_and_has_no_lower_bound():
    """Loads everything up to end_date (inclusive) with no start filter --
    signals need pre-window history to warm up. Would catch an accidental
    lower bound truncating that warm-up history."""
    conn = make_conn()
    insert_ohlcv(
        conn,
        "AAA",
        [("2024-01-01", 100.0, 101.0), ("2024-01-02", 102.0, 103.0), ("2024-01-05", 999.0, 999.0)],
    )
    result = _load_ohlcv_history(conn, ["AAA"], "2024-01-02")
    assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert set(result.columns) >= {"symbol", "date", "open", "high", "low", "close", "adj_close", "volume"}
    conn.close()


def test_load_forward_returns_selects_requested_horizon_column():
    """Would catch the wrong fwd_return_*d column being read (e.g. always
    fwd_return_1d regardless of the requested horizon)."""
    conn = make_conn()
    ensure_forward_returns_schema(conn)
    conn.execute(
        "INSERT INTO forward_returns (symbol, date, fwd_return_1d, fwd_return_5d) VALUES (?, ?, ?, ?)",
        ["AAA", "2024-01-01", 0.01, 0.09],
    )
    result = _load_forward_returns(conn, ["AAA"], "2024-01-01", "2024-01-01", "fwd_return_5d")
    assert result["fwd_return"].iloc[0] == pytest.approx(0.09)
    conn.close()


# ---------------------------------------------------------------------------
# screen_signal (end-to-end, synthetic panel)
# ---------------------------------------------------------------------------


def test_screen_signal_end_to_end_writes_outputs_and_verdict(tmp_path):
    """Seeds a small multi-symbol OHLCV panel with a strong, deliberately
    engineered momentum/forward-return relationship, computes forward
    returns, and runs the full screen_signal pipeline. Would catch a wiring
    bug across signal_library -> forward_returns -> ic_analysis ->
    decile_analysis -> screen.py's own result assembly that a narrower test
    of each piece in isolation could miss (e.g. a merge that silently drops
    all rows, always yielding an empty/degenerate result).
    """
    conn = make_conn()
    # 5 symbols, 30 days, momentum(window=5) strongly predicts fwd_return_1d
    # by construction: symbol i's daily return on day t is a fixed function
    # of i, so momentum ranks stay stable and correlate with next-day return.
    n_days = 30
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D").strftime("%Y-%m-%d").tolist()
    for i, symbol in enumerate(["A", "B", "C", "D", "E"]):
        daily_return = 0.001 * (i + 1)  # A drifts slowest, E fastest
        price = 100.0
        rows = []
        for date in dates:
            rows.append((date, price, price))
            price *= 1 + daily_return
        insert_ohlcv(conn, symbol, rows)

    symbols = ["A", "B", "C", "D", "E"]
    compute_and_store_forward_returns(conn, symbols=symbols)

    result = screen_signal(
        conn,
        signal_name="momentum",
        params={"window": 5},
        horizon="1d",
        start_date="2024-01-10",
        end_date="2024-01-28",
        symbols=symbols,
        method="spearman",
        output_dir=tmp_path,
    )

    assert result["signal"] == "momentum"
    assert result["horizon"] == "1d"
    assert "verdict" in result
    assert (tmp_path / "momentum_1d_ic.png").exists()
    assert (tmp_path / "momentum_1d_deciles.png").exists()
    assert result["n_days"] > 0
    # By construction momentum perfectly ranks the 5 symbols every day the
    # same way -> IC should be strongly positive, not just nonzero.
    assert result["mean_ic"] > 0.5


def test_comparison_table_sorts_by_ic_ir_descending_with_nan_last():
    """Would catch NaN IC-IR rows (e.g. a zero-variance signal) sorting to
    the top instead of the bottom, or the sort direction being ascending."""
    results = [
        {"signal": "low", "mean_ic": 0.01, "ic_ir": 0.5, "t_stat": 1.0, "decile_spread": 0.001, "verdict": "x"},
        {"signal": "high", "mean_ic": 0.05, "ic_ir": 2.0, "t_stat": 3.0, "decile_spread": 0.01, "verdict": "y"},
        {"signal": "nan", "mean_ic": float("nan"), "ic_ir": float("nan"), "t_stat": float("nan"), "decile_spread": float("nan"), "verdict": "z"},
    ]
    table = _comparison_table(results)
    assert table["signal"].tolist() == ["high", "low", "nan"]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_build_parser_run_command_defaults_and_overrides():
    """Would catch a default value drifting from what's documented, or a
    required flag silently becoming optional."""
    parser = _build_parser()
    args = parser.parse_args(["run", "--signal", "momentum"])
    assert args.command == "run"
    assert args.signal == "momentum"
    assert args.horizon == "5d"
    assert args.universe == "nifty50"
    assert args.method == "spearman"

    args2 = parser.parse_args(
        ["run", "--signal", "rsi_level", "--horizon", "20d", "--method", "pearson", "--universe", "RELIANCE,TCS"]
    )
    assert args2.horizon == "20d"
    assert args2.method == "pearson"
    assert args2.universe == "RELIANCE,TCS"


def test_build_parser_rejects_invalid_horizon_and_method():
    """--horizon/--method are constrained by `choices=` -- would catch that
    constraint being dropped, silently accepting a nonsense horizon string."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--signal", "momentum", "--horizon", "3d"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--signal", "momentum", "--method", "kendall"])


def test_build_parser_run_requires_signal_batch_requires_signals():
    """Would catch --signal/--signals being accidentally made optional."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])
    with pytest.raises(SystemExit):
        parser.parse_args(["batch"])

    args = parser.parse_args(["batch", "--signals", "momentum,rsi_level"])
    assert args.signals == "momentum,rsi_level"

"""Tests for backtest_cli.py's pure logic: CLI param parsing, strategy
construction/validation, date-range resolution, argparse wiring, and the
run-summary load/save round trip.

The three subcommand handlers that connect directly to
`backtest_cli.DEFAULT_DB_PATH` (`_run_command`, `_show_results_command`,
`_compare_command`) are intentionally out of scope here -- they hardcode a
real `duckdb.connect(str(DEFAULT_DB_PATH))` rather than accepting a
connection, so exercising them end-to-end would mean monkeypatching that
module-level path to a seeded temp file. The pieces they're built from
(`_build_strategy`, `_resolve_date_range`, `_load_run_summary`,
`_save_run_summary`, `_parse_params`) are each tested directly below instead.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import pytest

import backtest_cli as cli
from backtest import ensure_backtest_schema
from tests.helpers import insert_ohlcv, make_conn


def test_parse_params_coerces_int_then_float_then_string():
    """Would catch: a coercion order bug (e.g. float() accepting "10" as 10.0
    when an int was intended), or a broken split/strip on messy CLI input.
    """
    result = cli._parse_params("fast_window=10,slow_window=30,exit_mode=trailing, note = hello ")
    assert result == {
        "fast_window": 10,
        "slow_window": 30,
        "exit_mode": "trailing",
        "note": "hello",
    }


def test_parse_params_float_value():
    assert cli._parse_params("num_std=2.5") == {"num_std": 2.5}


@pytest.mark.parametrize("empty", [None, ""])
def test_parse_params_empty_input_returns_empty_dict(empty):
    assert cli._parse_params(empty) == {}


def test_parse_params_skips_pair_with_no_key():
    """A stray comma or leading '=value' shouldn't produce a bogus empty-string key."""
    assert cli._parse_params(",fast_window=10,=5") == {"fast_window": 10}


def test_build_strategy_unknown_name_prints_error_and_returns_none(capsys):
    """Would catch: raising instead of the documented None-return-plus-stderr-message contract."""
    result = cli._build_strategy("not_a_real_strategy", {})
    assert result is None
    captured = capsys.readouterr()
    assert "Unknown strategy" in captured.err


def test_build_strategy_unknown_param_returns_none(capsys):
    """Would catch: silently ignoring a typo'd parameter name instead of rejecting it."""
    result = cli._build_strategy("sma_crossover", {"fast_windoww": 10})
    assert result is None
    captured = capsys.readouterr()
    assert "unknown parameter" in captured.err.lower()
    assert "fast_windoww" in captured.err


def test_build_strategy_invalid_param_combo_returns_none(capsys):
    """sma_crossover's own validate() rejects fast_window >= slow_window --
    _build_strategy must surface that as a None + stderr message, not a raised exception.
    """
    result = cli._build_strategy("sma_crossover", {"fast_window": 50, "slow_window": 20})
    assert result is None
    captured = capsys.readouterr()
    assert "invalid parameters" in captured.err.lower()


def test_build_strategy_valid_construction():
    strategy = cli._build_strategy("sma_crossover", {"fast_window": 10, "slow_window": 30})
    assert strategy is not None
    assert strategy.config.fast_window == 10
    assert strategy.config.slow_window == 30


def test_build_strategy_defaults_when_no_params_given():
    strategy = cli._build_strategy("sma_crossover", {})
    assert strategy is not None
    assert strategy.config.fast_window == 20  # SmaCrossoverConfig's documented default
    assert strategy.config.slow_window == 50


def test_resolve_date_range_explicit_args_pass_through_without_querying_db():
    """When both --start and --end are given, _resolve_date_range must not
    need any data in the DB at all (would catch it querying/crashing anyway)."""
    conn = make_conn()  # empty ohlcv_data
    start, end = cli._resolve_date_range(conn, ["ANY"], "2024-01-01", "2024-06-30")
    assert (start, end) == ("2024-01-01", "2024-06-30")


def test_resolve_date_range_defaults_to_full_available_range():
    conn = make_conn()
    insert_ohlcv(
        conn,
        "RANGECO",
        [("2024-02-01", 10, 10), ("2024-02-15", 11, 11), ("2024-03-01", 12, 12)],
    )
    start, end = cli._resolve_date_range(conn, ["RANGECO"], None, None)
    assert start == "2024-02-01"
    assert end == "2024-03-01"


def test_resolve_date_range_partial_override_keeps_the_other_default():
    conn = make_conn()
    insert_ohlcv(
        conn, "RANGECO", [("2024-02-01", 10, 10), ("2024-03-01", 12, 12)]
    )
    start, end = cli._resolve_date_range(conn, ["RANGECO"], "2024-02-15", None)
    assert start == "2024-02-15"
    assert end == "2024-03-01"


def test_resolve_date_range_no_data_raises_runtime_error():
    conn = make_conn()
    with pytest.raises(RuntimeError):
        cli._resolve_date_range(conn, ["NODATA"], None, None)


def test_first_doc_line_returns_first_line_only():
    class Documented:
        """First line.

        More detail that should be excluded.
        """

    assert cli._first_doc_line(Documented) == "First line."


def test_first_doc_line_placeholder_when_no_docstring():
    class Undocumented:
        pass

    assert cli._first_doc_line(Undocumented) == "(no description)"


def test_strategy_not_found_message_lists_available_strategies():
    message = cli._strategy_not_found_message("nonexistent_strategy")
    assert "nonexistent_strategy" in message
    assert "sma_crossover" in message  # a real registered strategy should be listed


def test_list_strategies_command_needs_no_db(capsys):
    """`list-strategies` only reads the in-process registry -- would catch an
    accidental DB connection being required for a command that shouldn't need one."""
    exit_code = cli._list_strategies_command(argparse.Namespace())
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "sma_crossover" in captured.out


def test_list_params_command_unknown_strategy_returns_1(capsys):
    exit_code = cli._list_params_command(argparse.Namespace(strategy="nope"))
    assert exit_code == 1
    assert "Unknown strategy" in capsys.readouterr().err


def test_list_params_command_known_strategy_shows_defaults(capsys):
    exit_code = cli._list_params_command(argparse.Namespace(strategy="sma_crossover"))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "fast_window" in out
    assert "default=20" in out


def _seed_backtest_run(conn, run_id="run-1", strategy_name="sma_crossover_10_30"):
    ensure_backtest_schema(conn)
    conn.execute(
        """
        INSERT INTO backtest_runs (run_id, strategy_name, start_date, end_date, initial_capital, position_sizing)
        VALUES (?, ?, '2024-01-01', '2024-12-31', 1000000, 'equal_weight')
        """,
        [run_id, strategy_name],
    )
    conn.execute(
        """
        INSERT INTO backtest_results (run_id, total_trades, win_rate, total_return_pct, cagr, max_drawdown_pct, sharpe_ratio, final_equity)
        VALUES (?, 2, 0.5, 12.3, 11.0, 5.5, 1.2, 1123000)
        """,
        [run_id],
    )
    conn.execute(
        """
        INSERT INTO backtest_trades (
            trade_id, run_id, symbol, signal_date, entry_date, entry_price, quantity,
            entry_cost, exit_date, exit_price, exit_cost, exit_reason, gross_pnl, net_pnl
        ) VALUES ('t1', ?, 'RELIANCE', '2024-01-02', '2024-01-03', 100.0, 10, 0.1, '2024-01-10', 110.0, 0.1, 'SIGNAL', 100.0, 99.8)
        """,
        [run_id],
    )


def test_load_run_summary_missing_run_id_returns_none():
    conn = make_conn()
    ensure_backtest_schema(conn)
    assert cli._load_run_summary(conn, "does-not-exist") is None


def test_load_run_summary_recovers_symbols_from_trades():
    """Would catch: symbol scope not being (correctly) derived from
    backtest_trades when backtest_runs itself doesn't store it."""
    conn = make_conn()
    _seed_backtest_run(conn)
    result = cli._load_run_summary(conn, "run-1")
    assert result is not None
    assert result["symbols"] == ["RELIANCE"]
    assert result["symbols_recoverable"] is True
    assert result["metrics"]["total_trades"] == 2


def test_load_run_summary_zero_trades_symbols_not_recoverable():
    """A run with metrics but zero closed trades has no way to recover which
    symbols were in scope -- must be reported as such, not as an empty-but-recoverable list."""
    conn = make_conn()
    ensure_backtest_schema(conn)
    conn.execute(
        """
        INSERT INTO backtest_runs (run_id, strategy_name, start_date, end_date, initial_capital, position_sizing)
        VALUES ('run-2', 'sma_crossover_10_30', '2024-01-01', '2024-12-31', 1000000, 'equal_weight')
        """
    )
    conn.execute(
        """
        INSERT INTO backtest_results (run_id, total_trades, win_rate, total_return_pct, cagr, max_drawdown_pct, sharpe_ratio, final_equity)
        VALUES ('run-2', 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000)
        """
    )
    result = cli._load_run_summary(conn, "run-2")
    assert result["symbols"] == []
    assert result["symbols_recoverable"] is False


def test_save_run_summary_round_trips_through_json(tmp_path):
    """Would catch: a non-JSON-serializable field (e.g. a raw Timestamp/UUID
    object) breaking json.dumps, or the saved file losing/renaming a field."""
    conn = make_conn()
    _seed_backtest_run(conn)
    result = cli._load_run_summary(conn, "run-1")

    saved_path = cli._save_run_summary(result, {"fast_window": 10, "slow_window": 30}, tmp_path)
    assert saved_path.exists()

    payload = json.loads(saved_path.read_text())
    assert payload["run_id"] == "run-1"
    assert payload["params"] == {"fast_window": 10, "slow_window": 30}
    assert payload["symbols"] == ["RELIANCE"]
    assert len(payload["trades"]) == 1
    assert payload["trades"][0]["symbol"] == "RELIANCE"


def test_build_parser_run_defaults():
    """Would catch: a changed/typo'd flag name or default value silently
    breaking every documented `backtest_cli run ...` usage example."""
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--strategy", "sma_crossover"])
    assert args.command == "run"
    assert args.strategy == "sma_crossover"
    assert args.params is None
    assert args.symbol is None
    assert args.universe == "nifty50"
    assert args.initial_capital == 1_000_000
    assert args.position_sizing == "equal_weight"
    assert args.slippage_pct == 0.05
    assert args.max_positions == 10


def test_build_parser_show_results_requires_run_id_or_latest():
    """--run-id and --latest are a mutually exclusive *required* group --
    omitting both must fail argparse validation (SystemExit), not proceed with
    both None."""
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["show-results"])


def test_build_parser_show_results_latest_with_strategy():
    parser = cli._build_parser()
    args = parser.parse_args(["show-results", "--latest", "--strategy", "sma_crossover_20_50"])
    assert args.latest is True
    assert args.run_id is None
    assert args.strategy == "sma_crossover_20_50"


def test_build_parser_compare_requires_strategy_or_run_ids():
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["compare"])


def test_build_parser_missing_command_exits():
    """No subcommand at all must fail (command is required) rather than main()
    later raising a confusing KeyError on args.command."""
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_main_dispatches_to_list_strategies(capsys):
    exit_code = cli.main(["list-strategies"])
    assert exit_code == 0
    assert "sma_crossover" in capsys.readouterr().out


def test_main_dispatches_to_list_params_with_unknown_strategy(capsys):
    exit_code = cli.main(["list-params", "--strategy", "nope"])
    assert exit_code == 1

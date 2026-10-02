"""Correctness tests for src/indicators.py against hand-computable synthetic data.

Every indicator test uses either a perfectly flat price series (trivial,
exact expected values for SMA/EMA/BB/volatility) or a short monotonic/
integer-ramp series whose rolling mean, RSI, or EMA can be worked out by an
independent formula recomputation in the test itself -- not just eyeballed
for plausibility. Each test's docstring says what specific bug it would
catch if it failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import indicators
from src.indicators import (
    _compute_rsi_14,
    _compute_symbol_indicators,
    _load_daily_ohlcv,
    _normalize_ohlcv_input,
    _prepare_indicator_staging,
    _resolve_symbols,
    compute_and_store_all,
    compute_indicators,
    ensure_indicators_schema,
    store_indicators,
)
from tests.helpers import insert_ohlcv, make_conn


def _series_df(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """A minimal symbol/date/adj_close frame -- all _compute_symbol_indicators
    and _compute_rsi_14 actually read (open/high/low/close/volume are unused
    by those two functions)."""
    dates = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "adj_close": closes})


# --------------------------------------------------------------------------
# _compute_rsi_14
# --------------------------------------------------------------------------


def test_rsi14_flat_series_is_defined_as_100():
    """A perfectly flat price has zero gains AND zero losses every day (0/0
    for the raw RS ratio) -- the implementation's `.where(avg_loss != 0,
    100.0)` guard defines this degenerate case as RSI=100, not NaN or 50.

    Would catch: a change to the zero-loss guard that leaves flat markets as
    NaN (crashing downstream consumers) or at some other value.
    """
    adj_close = pd.Series([100.0] * 20)
    rsi = _compute_rsi_14(adj_close)
    # First valid average needs 14 non-null diffs; diff()'s own first value
    # (index 0) is NaN, so the first fully-populated 14-window ends at
    # index 14, not 13.
    assert rsi.iloc[14:].notna().all()
    assert (rsi.iloc[14:] == 100.0).all()


def test_rsi14_strictly_increasing_series_is_exactly_100():
    """Zero losses (every day is a gain) -> RS is undefined-but-guarded to
    100, same guard as the flat case but via a genuinely nonzero avg_gain.

    Would catch: an RSI formula that only special-cases the flat/0-vs-0 case
    and mishandles nonzero-gain/zero-loss (e.g. producing inf or NaN instead
    of 100).
    """
    adj_close = pd.Series(range(1, 21), dtype=float)  # +1 every day, no losses ever
    rsi = _compute_rsi_14(adj_close)
    assert rsi.iloc[14:].notna().all()
    assert (rsi.iloc[14:] == 100.0).all()


def test_rsi14_strictly_decreasing_series_is_exactly_0():
    """Zero gains (every day is a loss) -> avg_gain=0 -> RS=0 -> RSI = 100 -
    100/(1+0) = 0 exactly.

    Would catch: the opposite-direction mirror of the zero-loss guard being
    missing or wrong (e.g. RSI incorrectly landing at 100 for a straight-down
    move because of a sign error).
    """
    adj_close = pd.Series(range(20, 0, -1), dtype=float)  # -1 every day, no gains ever
    rsi = _compute_rsi_14(adj_close)
    assert rsi.iloc[14:].notna().all()
    assert (rsi.iloc[14:] == 0.0).all()


def test_rsi14_matches_independent_formula_on_mixed_series():
    """A mixed up/down series checked against an independently recomputed
    version of the exact documented formula (avg gain/loss over 14 bars,
    RS, 100-100/(1+RS)).

    Would catch: any deviation from the documented formula for a genuinely
    non-degenerate case -- e.g. using a different averaging window, an
    off-by-one in the rolling alignment, or Wilder's smoothing instead of a
    plain rolling mean.
    """
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, size=30))
    adj_close = pd.Series(prices)

    change = adj_close.diff()
    gain = change.clip(lower=0)
    loss = (-change).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    expected = 100.0 - (100.0 / (1.0 + rs))
    expected = expected.where(avg_loss != 0, 100.0)
    expected = expected.where(avg_gain.notna() & avg_loss.notna())

    actual = _compute_rsi_14(adj_close)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_rsi14_insufficient_history_is_all_nan_not_a_crash():
    """Fewer than 14 price changes (14 rows -> 13 diffs) -- rolling
    min_periods=14 means the rolling window can never fill, so RSI must be
    all-NaN, not raise.

    Would catch: an off-by-one in min_periods that either produces a
    (wrong) value too early or raises on short history instead of
    warming up with NaN.
    """
    adj_close = pd.Series(range(1, 14), dtype=float)  # 13 rows -> only 12 diffs
    rsi = _compute_rsi_14(adj_close)
    assert rsi.isna().all()


# --------------------------------------------------------------------------
# _compute_symbol_indicators
# --------------------------------------------------------------------------


def test_symbol_indicators_flat_series_exact_values():
    """On a perfectly flat 60-day price, once warmed every indicator has an
    exact, trivially hand-checkable value: sma_20/sma_50/ema_12/ema_26 all
    equal the flat price, daily_return is 0, bb_std is 0, and bb_upper ==
    bb_middle == bb_lower.

    Would catch: an indicator formula that drifts away from a perfectly flat
    input (e.g. a bug in the EWM recursion, or Bollinger bands not
    collapsing to a single value when volatility is genuinely zero).
    """
    df = _series_df("FLATCO", [100.0] * 60)
    out = _compute_symbol_indicators(df)

    warmed = out.iloc[59:]  # sma_50 needs 50 rows -> fully warmed by row index 49; use tail for safety
    assert (warmed["sma_20"] == 100.0).all()
    assert (warmed["sma_50"] == 100.0).all()
    assert (warmed["ema_12"] == 100.0).all()
    assert (warmed["ema_26"] == 100.0).all()
    assert (out["daily_return"].iloc[1:] == 0.0).all()
    assert (warmed["bb_std"] == 0.0).all()
    assert (warmed["bb_middle"] == 100.0).all()
    assert (warmed["bb_upper"] == 100.0).all()
    assert (warmed["bb_lower"] == 100.0).all()


def test_symbol_indicators_sma20_rolling_mean_of_integer_ramp():
    """sma_20 on a known 1..25 integer ramp: the mean of N consecutive
    integers is (first+last)/2, so each warmed row's expected value is
    hand-computable without a reference implementation.

    Would catch: an off-by-one in the rolling window (e.g. window=20 vs 21,
    or including/excluding the current row incorrectly).
    """
    closes = list(range(1, 26))  # 1..25
    df = _series_df("RAMPCO", [float(c) for c in closes])
    out = _compute_symbol_indicators(df)

    # First valid sma_20 is at row index 19 (20th row): mean(1..20) = 10.5.
    assert out["sma_20"].iloc[:19].isna().all()
    assert out["sma_20"].iloc[19] == pytest.approx(10.5)
    # Row index 24 (25th row): mean(6..25) = 15.5.
    assert out["sma_20"].iloc[24] == pytest.approx(15.5)
    # sma_50 never warms up with only 25 rows.
    assert out["sma_50"].isna().all()
    # ema_26 never warms up with only 25 rows (min_periods=26).
    assert out["ema_26"].isna().all()


def test_symbol_indicators_ema12_matches_independent_recursive_formula():
    """ema_12 on a short varying series, checked against an independently
    recomputed recursive EWM (alpha=2/13, adjust=False) -- confirms the
    *entire* history from row 0 feeds the recursion (min_periods only delays
    when a value is reported, it doesn't restart the recursion at the
    min_periods'th row).

    Would catch: a bug that computes EMA as if it were a simple moving
    average of only the last 12 rows, rather than pandas' full recursive EWM.
    """
    closes = [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 110.0, 108.0, 111.0, 115.0, 113.0, 116.0, 120.0]
    df = _series_df("EMACO", closes)
    out = _compute_symbol_indicators(df)

    alpha = 2.0 / (12 + 1)
    expected = closes[0]
    expected_series = [expected]
    for price in closes[1:]:
        expected = alpha * price + (1 - alpha) * expected
        expected_series.append(expected)

    assert out["ema_12"].iloc[:11].isna().all()
    for i in range(11, len(closes)):
        assert out["ema_12"].iloc[i] == pytest.approx(expected_series[i])


def test_symbol_indicators_volatility20_needs_21_price_rows_not_20():
    """daily_return's own first value is NaN (pct_change of the very first
    row has nothing to diff against), so a rolling std with window=20,
    min_periods=20 over daily_return can't produce its first non-NaN value
    until the window no longer straddles that initial NaN -- i.e. at price
    row index 20 (the 21st row), not row index 19.

    Would catch: an off-by-one that assumes volatility_20 warms up after 20
    price rows (it actually needs 21), which would silently misalign every
    downstream consumer of "the first N valid volatility rows."
    """
    df = _series_df("VOLCO", [100.0 + i for i in range(25)])
    out = _compute_symbol_indicators(df)

    assert out["volatility_20"].iloc[:20].isna().all()
    assert pd.notna(out["volatility_20"].iloc[20])


def test_symbol_indicators_bollinger_bands_widen_with_volatility():
    """bb_upper/bb_lower must straddle bb_middle by exactly 2*bb_std on both
    sides (BB_NUM_STD=2), for a genuinely non-zero-volatility series.

    Would catch: a wrong band multiplier, or bb_upper/bb_lower swapped.
    """
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 2, size=25))
    df = _series_df("BANDCO", list(closes))
    out = _compute_symbol_indicators(df)

    warmed = out.iloc[19:]
    np.testing.assert_allclose(
        (warmed["bb_upper"] - warmed["bb_middle"]).to_numpy(), (2.0 * warmed["bb_std"]).to_numpy()
    )
    np.testing.assert_allclose(
        (warmed["bb_middle"] - warmed["bb_lower"]).to_numpy(), (2.0 * warmed["bb_std"]).to_numpy()
    )
    assert (warmed["bb_std"] > 0).all()
    assert (warmed["bb_upper"] > warmed["bb_lower"]).all()


# --------------------------------------------------------------------------
# _normalize_ohlcv_input
# --------------------------------------------------------------------------


def test_normalize_ohlcv_input_missing_column_raises_with_names():
    """Would catch: a missing-column check that silently drops/ignores the
    problem instead of failing loudly, or that reports the wrong column
    names.
    """
    df = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"], "open": [1.0]})
    with pytest.raises(ValueError, match="high"):
        _normalize_ohlcv_input(df)


def test_normalize_ohlcv_input_falls_back_to_timestamp_column():
    """When 'date' is absent but 'timestamp' is present, normalize should
    derive 'date' from it (matching how ohlcv_data itself is shaped).

    Would catch: a regression that requires callers to pre-rename
    'timestamp' to 'date' before calling compute_indicators.
    """
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-01"]),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "adj_close": [1.0, 1.0],
            "volume": [0, 0],
        }
    )
    out = _normalize_ohlcv_input(df)
    assert "date" in out.columns
    # Also verify the sort-by-symbol-then-date happened.
    assert list(out["date"]) == sorted(out["date"])


def test_normalize_ohlcv_input_empty_returns_empty():
    """Would catch: an empty-input crash (e.g. groupby on an empty frame
    raising instead of being handled before reaching that point)."""
    df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    out = _normalize_ohlcv_input(df)
    assert out.empty


# --------------------------------------------------------------------------
# compute_indicators: multi-symbol isolation and empty input
# --------------------------------------------------------------------------


def test_compute_indicators_does_not_leak_across_symbols():
    """Two symbols at wildly different, individually-flat price levels,
    inserted pre-sorted by date first (interleaved across symbols) rather
    than grouped by symbol, to make sure per-symbol grouping -- not row
    order -- determines each rolling window's inputs.

    Would catch: a rolling/EWM computation applied to the whole DataFrame
    before splitting by symbol (or a groupby that doesn't actually isolate
    each group), which would blend one symbol's price level into another's
    indicator values at the seam between symbols.
    """
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    rows = []
    for d in dates:
        rows.append({"symbol": "ZERO", "date": d, "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "adj_close": 0.0, "volume": 0})
        rows.append({"symbol": "THOUSAND", "date": d, "open": 1000.0, "high": 1000.0, "low": 1000.0, "close": 1000.0, "adj_close": 1000.0, "volume": 0})
    df = pd.DataFrame(rows)

    out = compute_indicators(df)
    zero_rows = out.loc[out["symbol"] == "ZERO"].sort_values("date")
    thousand_rows = out.loc[out["symbol"] == "THOUSAND"].sort_values("date")

    assert (zero_rows["sma_20"].dropna() == 0.0).all()
    assert (thousand_rows["sma_20"].dropna() == 1000.0).all()
    assert len(zero_rows) == 25
    assert len(thousand_rows) == 25


def test_compute_indicators_empty_input_returns_expected_columns():
    """Would catch: an empty-input path that returns the wrong column set
    (breaking a caller that does `indicators[INDICATOR_OUTPUT_COLUMNS]`
    downstream) or raises instead of returning an empty frame.
    """
    df = pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"])
    out = compute_indicators(df)
    assert out.empty
    assert list(out.columns) == list(indicators.INDICATOR_OUTPUT_COLUMNS)


# --------------------------------------------------------------------------
# _prepare_indicator_staging
# --------------------------------------------------------------------------


def test_prepare_indicator_staging_missing_column_raises():
    df = pd.DataFrame({"symbol": ["A"], "date": ["2024-01-01"]})
    with pytest.raises(ValueError, match="sma_20"):
        _prepare_indicator_staging(df)


def test_prepare_indicator_staging_attaches_computed_at():
    df = pd.DataFrame(
        {col: [0.0] for col in indicators.INDICATOR_OUTPUT_COLUMNS if col not in ("symbol", "date")}
    )
    df.insert(0, "date", ["2024-01-01"])
    df.insert(0, "symbol", ["A"])
    staging = _prepare_indicator_staging(df)
    assert "computed_at" in staging.columns
    assert pd.notna(staging["computed_at"].iloc[0])


# --------------------------------------------------------------------------
# store_indicators: insert vs upsert-update, empty short-circuit
# --------------------------------------------------------------------------


def test_store_indicators_empty_df_short_circuits_without_creating_schema():
    """Would catch: calling ensure_indicators_schema (or otherwise touching
    the DB) unconditionally on an empty DataFrame, or the zero-summary shape
    changing.
    """
    conn = make_conn()
    summary = store_indicators(conn, pd.DataFrame())
    assert summary == {"rows_inserted": 0, "rows_updated": 0, "total_rows": 0}
    tables = conn.execute("SELECT table_name FROM information_schema.tables").df()["table_name"].tolist()
    assert "indicators_daily" not in tables
    conn.close()


def test_store_indicators_insert_then_upsert_updates_existing_row():
    """First store is a pure insert (rows_inserted == total_rows,
    rows_updated == 0); storing again for the same (symbol, date) with a
    changed value must update in place (rows_updated == total_rows,
    rows_inserted == 0) and the stored value must reflect the new one, not
    the old one.

    Would catch: an upsert that silently double-inserts (violating the
    (symbol, date) primary key or duplicating rows) or that miscounts
    inserted vs. updated rows.
    """
    conn = make_conn()
    base_row = {col: [np.nan] for col in indicators.INDICATOR_OUTPUT_COLUMNS}
    base_row["symbol"] = ["A"]
    base_row["date"] = ["2024-01-01"]
    base_row["sma_20"] = [10.0]
    df = pd.DataFrame(base_row)

    first = store_indicators(conn, df)
    assert first == {"rows_inserted": 1, "rows_updated": 0, "total_rows": 1}

    df.loc[0, "sma_20"] = 99.0
    second = store_indicators(conn, df)
    assert second == {"rows_inserted": 0, "rows_updated": 1, "total_rows": 1}

    stored = conn.execute("SELECT sma_20 FROM indicators_daily WHERE symbol = 'A' AND date = '2024-01-01'").fetchone()
    assert stored[0] == pytest.approx(99.0)
    conn.close()


# --------------------------------------------------------------------------
# _load_daily_ohlcv: timeframe filtering and symbol filtering
# --------------------------------------------------------------------------


def test_load_daily_ohlcv_filters_by_timeframe():
    """A 1h row for the same symbol/date must never be returned by a daily
    indicator load.

    Would catch: a missing or wrong `timeframe = '1d'` filter that mixes
    intraday and daily bars into the same indicator computation.
    """
    conn = make_conn()
    insert_ohlcv(conn, "A", [("2024-01-01", 100.0, 101.0)], timeframe="1d")
    insert_ohlcv(conn, "A", [("2024-01-01", 999.0, 999.0)], timeframe="1h")
    out = _load_daily_ohlcv(conn)
    assert len(out) == 1
    assert out["open"].iloc[0] == pytest.approx(100.0)
    conn.close()


def test_load_daily_ohlcv_filters_by_symbol_list():
    conn = make_conn()
    insert_ohlcv(conn, "A", [("2024-01-01", 1.0, 1.0)])
    insert_ohlcv(conn, "B", [("2024-01-01", 2.0, 2.0)])
    out = _load_daily_ohlcv(conn, symbols=["A"])
    assert out["symbol"].tolist() == ["A"]
    conn.close()


# --------------------------------------------------------------------------
# _resolve_symbols: explicit list vs universe fallback
# --------------------------------------------------------------------------


def test_resolve_symbols_strips_ns_suffix_from_explicit_list():
    """Would catch: a symbol filter that forgets to strip '.NS' before
    matching against storage symbols (which never carry the suffix), which
    would silently return zero rows for every explicitly-requested symbol.
    """
    conn = make_conn()
    out = _resolve_symbols(conn, ["RELIANCE.NS", "TCS"])
    assert out == ["RELIANCE", "TCS"]
    conn.close()


def test_resolve_symbols_falls_back_to_active_universe(monkeypatch):
    """symbols=None must resolve via get_active_universe(), with the same
    '.NS'-stripping applied.

    Would catch: the universe fallback being skipped, or the suffix not
    being stripped on that path even though it is on the explicit-list path.
    """
    conn = make_conn()
    monkeypatch.setattr(indicators, "get_active_universe", lambda c: ["RELIANCE.NS", "TCS.NS"])
    out = _resolve_symbols(conn, None)
    assert out == ["RELIANCE", "TCS"]
    conn.close()


# --------------------------------------------------------------------------
# compute_and_store_all: end-to-end
# --------------------------------------------------------------------------


def test_compute_and_store_all_end_to_end_with_explicit_symbols():
    """Seeds ohlcv_data with a flat 25-day series for one symbol, runs the
    full compute+store pipeline, and checks both the summary dict and the
    persisted sma_20 value against the hand-computable flat-series
    expectation (sma_20 == the flat price once warmed).

    Would catch: any break in the wiring between _load_daily_ohlcv,
    compute_indicators, and store_indicators (e.g. a column dropped in
    transit, or the summary's ohlcv_rows/symbols counts not matching what
    was actually loaded).
    """
    conn = make_conn()
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    rows = [(d.strftime("%Y-%m-%d"), 50.0, 50.0) for d in dates]
    insert_ohlcv(conn, "FLATCO", rows)

    summary = compute_and_store_all(conn, symbols=["FLATCO"])
    assert summary["symbols"] == 1
    assert summary["ohlcv_rows"] == 25
    assert summary["rows_inserted"] == 25
    assert summary["rows_updated"] == 0
    assert summary["total_rows"] == 25

    stored = conn.execute(
        "SELECT sma_20 FROM indicators_daily WHERE symbol = 'FLATCO' ORDER BY date"
    ).df()
    assert stored["sma_20"].iloc[19:].eq(50.0).all()
    assert stored["sma_20"].iloc[:19].isna().all()
    conn.close()


def test_compute_and_store_all_no_matching_ohlcv_returns_zero_summary():
    """Would catch: an empty-OHLCV path that crashes (e.g. calling
    compute_indicators/store_indicators on an empty frame in a way that
    isn't actually guarded) instead of returning the documented all-zero
    summary.
    """
    conn = make_conn()
    summary = compute_and_store_all(conn, symbols=["NOPE"])
    assert summary == {
        "symbols": 0,
        "ohlcv_rows": 0,
        "rows_inserted": 0,
        "rows_updated": 0,
        "total_rows": 0,
    }
    conn.close()

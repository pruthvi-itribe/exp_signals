"""Light tests for dbplots.py's DB-query helpers.

dbplots.py is an exploratory/ad-hoc script (its own docstring: "update this"
placeholders for DB_PATH/column names) rather than a core pipeline module,
so this suite is intentionally light: it covers the parameterized-SQL query
builders with a small synthetic table, and smoke-tests the two plotting
functions (which call `plt.show()`, not a file save) with `plt.show`
monkeypatched to a no-op so they can run headless.
"""

from __future__ import annotations

import duckdb
import matplotlib

matplotlib.use("Agg")

import dbplots


def _make_price_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE ohlcv_data (
            symbol VARCHAR, timestamp TIMESTAMP, adj_close DOUBLE
        )
        """
    )
    conn.executemany(
        "INSERT INTO ohlcv_data VALUES (?, ?, ?)",
        [
            ("AAA", "2024-01-01", 100.0),
            ("AAA", "2024-01-02", 101.0),
            ("BBB", "2024-01-01", 200.0),
        ],
    )


def test_list_tables_returns_created_table():
    conn = duckdb.connect(":memory:")
    _make_price_table(conn)
    tables = dbplots.list_tables(conn)
    assert "ohlcv_data" in tables.iloc[:, 0].tolist()


def test_describe_table_lists_columns():
    conn = duckdb.connect(":memory:")
    _make_price_table(conn)
    described = dbplots.describe_table(conn, "ohlcv_data")
    assert set(described["column_name"]) == {"symbol", "timestamp", "adj_close"}


def test_list_commodities_returns_distinct_sorted_symbols():
    """Would catch a query that returns duplicates or the wrong ordering."""
    conn = duckdb.connect(":memory:")
    _make_price_table(conn)
    result = dbplots.list_commodities(conn, "ohlcv_data", "symbol")
    assert result.iloc[:, 0].tolist() == ["AAA", "BBB"]


def test_get_commodity_prices_filters_by_symbol_and_date_bounds():
    """Would catch the optional start_date/end_date filters being silently
    ignored, or one symbol's rows leaking into another's result."""
    conn = duckdb.connect(":memory:")
    _make_price_table(conn)
    result = dbplots.get_commodity_prices(
        conn, "ohlcv_data", "AAA", commodity_col="symbol", date_col="timestamp", price_col="adj_close"
    )
    assert len(result) == 2
    assert result["price"].tolist() == [100.0, 101.0]

    bounded = dbplots.get_commodity_prices(
        conn,
        "ohlcv_data",
        "AAA",
        commodity_col="symbol",
        date_col="timestamp",
        price_col="adj_close",
        start_date="2024-01-02",
    )
    assert len(bounded) == 1
    assert bounded["price"].iloc[0] == 101.0


def test_get_all_commodity_features_returns_only_that_symbols_rows():
    conn = duckdb.connect(":memory:")
    _make_price_table(conn)
    result = dbplots.get_all_commodity_features(conn, "ohlcv_data", "BBB", commodity_col="symbol", date_col="timestamp")
    assert len(result) == 1
    assert result["symbol"].iloc[0] == "BBB"


def test_plot_price_runs_headless(monkeypatch):
    """Smoke test only -- confirms the plotting call doesn't crash on valid
    input when a display isn't available (plt.show patched to a no-op)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    monkeypatch.setattr(plt, "show", lambda: None)
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price": [100.0, 101.0]})
    dbplots.plot_price(df, "AAA")


def test_plot_price_with_feature_runs_headless(monkeypatch):
    import matplotlib.pyplot as plt
    import pandas as pd

    monkeypatch.setattr(plt, "show", lambda: None)
    price_df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price": [100.0, 101.0]})
    feature_df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "price": [1.5, 1.6]})
    dbplots.plot_price_with_feature(price_df, feature_df, "AAA", "RSI")

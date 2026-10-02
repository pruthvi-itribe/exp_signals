"""
Explore a local DuckDB file and plot commodity prices over time.

Usage:
    python plot_commodity_prices.py
"""

import duckdb
import matplotlib.pyplot as plt

DB_PATH = "data/trading_data.duckdb"   # <-- update this
PRICE_TABLE = "ohlcv_data"          # <-- update once you know the table name
COMMODITY_COL = "symbol"        # <-- update to match your schema
DATE_COL = "timestamp"                  # <-- update to match your schema
PRICE_COL = "adj_close"                # <-- update to match your schema
FEATURES_TABLE = "indicators_daily"
FEATURES_TAB_DATE_COL = "date"


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


def list_tables(con: duckdb.DuckDBPyConnection):
    """Return all table names in the database."""
    return con.execute("SHOW TABLES").df()


def describe_table(con: duckdb.DuckDBPyConnection, table: str):
    """Return column names/types for a given table — handy for finding the right column names."""
    return con.execute(f"DESCRIBE {table}").df()


def list_commodities(con: duckdb.DuckDBPyConnection, table: str, commodity_col: str):
    """Return distinct commodity names available in the price table."""
    return con.execute(
        f"SELECT DISTINCT {commodity_col} FROM {table} ORDER BY {commodity_col}"
    ).df()


def get_commodity_prices(
    con: duckdb.DuckDBPyConnection,
    table: str,
    commodity: str,
    commodity_col: str = COMMODITY_COL,
    date_col: str = DATE_COL,
    price_col: str = PRICE_COL,
    start_date: str = None,
    end_date: str = None,
):
    """Fetch date/price rows for one commodity, optionally bounded by date."""
    query = f"""
        SELECT {date_col} AS date, {price_col} AS price
        FROM {table}
        WHERE {commodity_col} = ?
    """
    params = [commodity]

    if start_date:
        query += f" AND {date_col} >= ?"
        params.append(start_date)
    if end_date:
        query += f" AND {date_col} <= ?"
        params.append(end_date)

    query += f" ORDER BY {date_col}"

    return con.execute(query, params).df()

def get_all_commodity_features(
    con: duckdb.DuckDBPyConnection,
    table: str,
    commodity: str,
    commodity_col: str = COMMODITY_COL,
    date_col: str = DATE_COL,
):
    """
    Fetch all features/columns related to a particular commodity in given table 
    """
    query = f"""
        SELECT *
        FROM {table}
        WHERE {commodity_col} = ?
        ORDER BY {date_col}
    """
    print(query)
    return con.execute(query, [commodity]).df()

def plot_price(df, commodity: str):
    """Simple single-line price plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["price"], color="tab:blue", label="Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{commodity} Price Over Time")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_price_with_feature(price_df, feature_df, commodity: str, feature_name: str):
    """
    Plot price and a second feature on the same chart using a secondary y-axis.
    feature_df should have columns: date, value
    (For later, once you're pulling a second data series for the same commodity.)
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(price_df["date"], price_df["price"], color="tab:blue", label="Price")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(feature_df["date"], feature_df["price"], color="tab:orange", label=feature_name)
    ax2.set_ylabel(feature_name, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"{commodity}: Price vs {feature_name}")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    con = connect(DB_PATH)

    # Step 1: see what tables exist
    print(list_tables(con))

    # Step 2: once you know the price table name, inspect its columns
    #print(describe_table(con, PRICE_TABLE))
    print(describe_table(con, "signals"))

    # Step 3: see what commodities are available
    #print(list_commodities(con, FEATURES_TABLE, COMMODITY_COL))

    # Add1 - show all entries related to a particular commodity in a table
    features_df = get_all_commodity_features(con, "signals", "RELIANCE", COMMODITY_COL, "date")
    print(features_df)


    # Step 4: fetch and plot a specific commodity
    df = get_commodity_prices(con, PRICE_TABLE, commodity="RELIANCE")
    features_df = get_commodity_prices(con, FEATURES_TABLE,
        commodity="RELIANCE",
        commodity_col = COMMODITY_COL,
        date_col = FEATURES_TAB_DATE_COL,
        price_col =  "sma_20")
    #print(df)
    #print(features_df)
    #plot_price(df, "RELIANCE")
    plot_price_with_feature(df, features_df, "RELIANCE", "sma_20")

    con.close()

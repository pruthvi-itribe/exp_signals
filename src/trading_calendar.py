"""
Trading calendar utilities for detecting and logging missing trading days.

This module provides functions to identify gaps in historical data, log them in a database,
and mark them as resolved once backfilled.  It's designed to catch silent data gaps
(e.g., yfinance API hiccups, delisted/relisted stocks, network failures during a fetch)
before they corrupt indicator calculations or strategy signals downstream.

Output Interpretation:
- 1-2 missing days scattered around might be API flakiness worth re-fetching.
- Large contiguous blocks of missing data suggest a stock may have been suspended
  from trading or delisted during that period — worth checking manually before
  including it in the universe.
"""

import pandas as pd
import duckdb
from pandas_market_calendars import get_calendar

def get_nse_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get all valid NSE trading session dates within a given range.

    Falls back to 'XBOM' (BSE) if NSE isn't a supported calendar name.
    """
    try:
        calendar = get_calendar('NSE')
    except ImportError:
        try:
            calendar = get_calendar('XBOM')  # Fallback to BSE
        except Exception:
            raise ValueError("Neither NSE nor XBOM calendar found in pandas_market_calendars.")
    
    return calendar.valid_days(start_date=start_date, end_date=end_date)

def find_missing_trading_days(conn: duckdb.DuckDBPyConnection, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Find dates that are valid trading days but missing from the database.
    """
    expected_days = get_nse_trading_days(start_date, end_date).tz_localize(None).normalize()
    
    query = f"""
    SELECT DISTINCT timestamp::DATE AS date
    FROM ohlcv_data
    WHERE symbol = '{symbol}'
    AND timestamp BETWEEN '{start_date}' AND '{end_date}'
    """
    
    actual_days = conn.execute(query).fetchall()
    actual_days = pd.to_datetime([row[0] for row in actual_days]).normalize()
    missing_days = expected_days.difference(actual_days)
    
    if missing_days.empty:
        return pd.DataFrame()
    
    # Calculate days since the previous available date -- each missing day
    # gets its OWN anchor (the previous available date immediately before
    # it), not one anchor shared across every gap in the batch. A symbol can
    # have more than one separate gap in range, and each needs its own
    # "days since previous available" measurement.
    missing_days = missing_days.to_series()
    days_since_previous = pd.Series(index=missing_days.index, dtype="int64")
    for idx, missing_date in missing_days.items():
        previous_available_date = conn.execute(
            f"SELECT MAX(timestamp) FROM ohlcv_data WHERE symbol = '{symbol}' AND timestamp < '{missing_date}'"
        ).fetchone()[0]
        if previous_available_date is None:
            days_since_previous[idx] = (missing_date - pd.to_datetime(start_date)).days
        else:
            days_since_previous[idx] = (missing_date - pd.to_datetime(previous_available_date)).days

    missing_df = pd.DataFrame({'symbol': [symbol] * len(missing_days), 'missing_date': missing_days, 'days_since_previous_available_date': days_since_previous})
    return missing_df

def audit_universe(conn: duckdb.DuckDBPyConnection, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Run find_missing_trading_days() across a list of symbols and return a combined summary dataframe.
    """
    all_gaps = []
    for symbol in symbols:
        gaps = find_missing_trading_days(conn, symbol, start_date, end_date)
        if not gaps.empty:
            all_gaps.append(gaps)
    
    if not all_gaps:
        return pd.DataFrame()
    
    combined_gaps = pd.concat(all_gaps, ignore_index=True)
    combined_gaps = combined_gaps.sort_values(by=['symbol', 'missing_date'])
    
    # Print per-symbol summary
    for symbol in symbols:
        symbol_gaps = combined_gaps[combined_gaps['symbol'] == symbol]
        if not symbol_gaps.empty:
            print(f"Symbol: {symbol}, Missing Days: {len(symbol_gaps)}, Most Recent: {symbol_gaps['missing_date'].max()}")
    
    return combined_gaps

def log_gaps(conn: duckdb.DuckDBPyConnection, gaps_df: pd.DataFrame) -> None:
    """
    Insert newly found gaps into the data_gaps_log table, avoiding duplicates.
    """
    if gaps_df.empty:
        return
    
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS data_gaps_log (
        symbol VARCHAR NOT NULL,
        missing_date DATE NOT NULL,
        detected_at TIMESTAMP NOT NULL,
        resolved BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (symbol, missing_date)
    )
    """)
    
    for index, row in gaps_df.iterrows():
        try:
            conn.execute(f"""
            INSERT INTO data_gaps_log (symbol, missing_date, detected_at)
            VALUES ('{row['symbol']}', '{row['missing_date'].strftime('%Y-%m-%d')}', CURRENT_TIMESTAMP)
            ON CONFLICT (symbol, missing_date) DO NOTHING
            """)
        except Exception as e:
            print(f"Error logging gap for {row['symbol']} on {row['missing_date']}: {e}")

def mark_resolved(conn: duckdb.DuckDBPyConnection, symbol: str, date: str) -> None:
    """
    Mark a gap as resolved in the data_gaps_log table.
    """
    conn.execute(f"""
    UPDATE data_gaps_log
    SET resolved = TRUE
    WHERE symbol = '{symbol}' AND missing_date = '{date}'
    """)

if __name__ == "__main__":
    import datetime

    conn = duckdb.connect("data/trading_data.duckdb")
    symbols = ["RELIANCE", "TCS", "HDFCBANK"]
    end_date = datetime.date(2026, 8, 26)
    start_date = end_date - datetime.timedelta(days=365)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    gaps_df = audit_universe(conn, symbols, start_date_str, end_date_str)
    print("\nMissing Days Summary:")
    print(gaps_df)

    if not gaps_df.empty:
        log_gaps(conn, gaps_df)
        print("\nData gaps logged to data_gaps_log.")
    else:
        print("\nNo data gaps found.")

    conn.close()
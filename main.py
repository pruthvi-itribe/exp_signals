from src.storage.db_manager import DatabaseManager
from src.ingestion.historical_fetcher import HistoricalFetcher
from src.universe import get_active_universe


def main() -> None:
    db = DatabaseManager()
    fetcher = HistoricalFetcher(db_manager=db)

    with db.get_connection() as conn:
        tickers = get_active_universe(conn)

    if not tickers:
        raise RuntimeError(
            "No active universe tickers found. Run `python -m src.universe` to sync Nifty 50 first."
        )

    #print(f"Ingesting 1 year of daily data for {len(tickers)} active universe tickers...")

    #for ticker in tickers:
    #    fetcher.fetch_and_store(symbol=ticker, interval="1d", period="1y")

    # Verify data in DuckDB
    print("\n--- DuckDB Output Verification (RELIANCE) ---")
    reliance_data = db.fetch_candles(symbol="RELIANCE", timeframe="1d")
    print(reliance_data)

if __name__ == "__main__":
    main()
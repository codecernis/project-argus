"""
Data Ingestion Module.
Responsible for fetching and normalizing raw market data from external APIs.
"""

import logging
import pandas as pd
import yfinance as yf
from src import config

logger = logging.getLogger(__name__)

def fetch_market_data() -> pd.DataFrame | None:
    """
    Fetches OHLCV market data from Yahoo Finance.
    Flattens MultiIndex columns if present and saves a raw snapshot to disk.

    Returns:
        pd.DataFrame: The normalized raw market data, or None if the fetch fails.
    """
    logger.info("Fetching market data for %s over period: %s", config.SYMBOL, config.PERIOD)
    
    try:
        df = yf.download(
            tickers=config.SYMBOL, 
            period=config.PERIOD, 
            interval=config.INTERVAL,
            progress=False
        )
        
        if df.empty:
            logger.error("No data found for symbol: %s. Verify ticker and network connection.", config.SYMBOL)
            return None

        # Standardize MultiIndex columns to a flat 2D schema
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Save raw snapshot for auditability
        df.to_csv(config.RAW_DATA_PATH)
        logger.info("Successfully ingested %d rows. Raw data saved to %s", len(df), config.RAW_DATA_PATH)
        
        return df

    except Exception as e:
        logger.error("An error occurred during data ingestion: %s", str(e), exc_info=True)
        return None
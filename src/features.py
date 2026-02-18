"""
Feature Engineering Module.
Calculates technical indicators and generates the target labels for machine learning.
"""

import logging
import pandas as pd
import pandas_ta_classic as ta
from src import config

logger = logging.getLogger(__name__)

def generate_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Calculates technical indicators and appends the target label based on future price action.

    Args:
        df (pd.DataFrame): The raw OHLCV market data.

    Returns:
        pd.DataFrame: The processed dataframe containing features and target labels.
    """
    logger.info("Initiating feature engineering pipeline...")
    
    try:
        df_processed = df.copy()

        # 1. Technical Indicators (Feature Set)
        df_processed.ta.rsi(length=config.RSI_LENGTH, append=True)
        df_processed.ta.ema(length=config.EMA_LENGTH, append=True)
        df_processed.ta.bbands(length=config.BB_LENGTH, append=True)
        df_processed.ta.macd(fast=12, slow=26, signal=9, append=True)
        df_processed.ta.atr(length=14, append=True)

        # 2. Target Labeling (1 for price increase, 0 for price decrease/flat)
        logger.info("Applying target labels with a %d-period lookahead.", config.LOOKAHEAD_PERIODS)
        future_close = df_processed['Close'].shift(-config.LOOKAHEAD_PERIODS)
        df_processed['Target'] = (future_close > df_processed['Close']).astype(int)

        # 3. Data Cleansing (Remove NaN values resulting from indicator warm-up and shifting)
        initial_row_count = len(df_processed)
        df_processed.dropna(inplace=True)
        dropped_rows = initial_row_count - len(df_processed)
        logger.info("Dropped %d incomplete rows (indicator warm-up and lookahead shift).", dropped_rows)

        # 4. Persistence
        df_processed.to_csv(config.PROCESSED_DATA_PATH)
        logger.info("Feature engineering complete. Processed data saved to %s", config.PROCESSED_DATA_PATH)

        return df_processed

    except Exception as e:
        logger.error("Failed to generate features: %s", str(e), exc_info=True)
        return None
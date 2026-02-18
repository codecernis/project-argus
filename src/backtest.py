"""
Backtesting Module.
Simulates trading execution based on model predictions, applies transaction costs,
and calculates institutional-grade risk metrics (Sharpe Ratio, Max Drawdown).
"""

import logging
import numpy as np
import pandas as pd
from src import config

logger = logging.getLogger(__name__)

def run_backtest(test_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Executes a vectorized historical backtest evaluating the model's performance 
    against a Buy & Hold benchmark.

    Args:
        test_df (pd.DataFrame): Dataframe containing historical prices and model predictions.

    Returns:
        pd.DataFrame: Dataframe containing the continuous equity curve and trade logs.
    """
    logger.info("Simulating strategy execution and calculating risk metrics...")

    try:
        # Calculate Asset Returns (Daily shift ensures no look-ahead bias)
        test_df['Daily_Return'] = test_df['Close'].pct_change().shift(-1)
        test_df.dropna(subset=['Daily_Return'], inplace=True)

        # Strategy Logic & Position Sizing
        test_df['Strategy_Return'] = 0.0
        
        # Allocate capital to the asset when prediction is 1 (Long)
        long_positions = test_df['Prediction'] == 1
        test_df.loc[long_positions, 'Strategy_Return'] = test_df.loc[long_positions, 'Daily_Return']

        # Transaction Costs 
        state_changes = (test_df['Prediction'].diff() != 0) & (test_df['Prediction'].notna())
        test_df.loc[state_changes, 'Strategy_Return'] -= config.TRADING_FEE

        # Equity Curve Calculation (Cumulative Product)
        test_df['Argus_Balance'] = config.STARTING_BALANCE * (1 + test_df['Strategy_Return']).cumprod()
        test_df['Buy_Hold_Balance'] = config.STARTING_BALANCE * (1 + test_df['Daily_Return']).cumprod()

        # Risk Metrics: Maximum Drawdown
        test_df['Argus_Peak'] = test_df['Argus_Balance'].cummax()
        argus_mdd = ((test_df['Argus_Balance'] - test_df['Argus_Peak']) / test_df['Argus_Peak']).min()

        test_df['BH_Peak'] = test_df['Buy_Hold_Balance'].cummax()
        bh_mdd = ((test_df['Buy_Hold_Balance'] - test_df['BH_Peak']) / test_df['BH_Peak']).min()

        # Risk Metrics: Annualized Sharpe Ratio
        def calculate_sharpe(returns: pd.Series, periods_per_year: int) -> float:
            if returns.std() == 0:
                return 0.0
            return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

        argus_sharpe = calculate_sharpe(test_df['Strategy_Return'], config.TRADING_DAYS_PER_YEAR)
        bh_sharpe = calculate_sharpe(test_df['Daily_Return'], config.TRADING_DAYS_PER_YEAR)

        # Reporting
        total_round_trips = state_changes.sum() // 2

        logger.info("--- DECADE WFO RESULTS (Starting Capital: $%.2f) ---", config.STARTING_BALANCE)
        
        logger.info("[ARGUS SYSTEM]")
        logger.info("  Final Balance : $%.2f", test_df['Argus_Balance'].iloc[-1])
        logger.info("  Round Trips   : %d", total_round_trips)
        logger.info("  Max Drawdown  : %.2f%%", argus_mdd * 100)
        logger.info("  Sharpe Ratio  : %.2f", argus_sharpe)
        
        logger.info("[BUY & HOLD BENCHMARK]")
        logger.info("  Final Balance : $%.2f", test_df['Buy_Hold_Balance'].iloc[-1])
        logger.info("  Max Drawdown  : %.2f%%", bh_mdd * 100)
        logger.info("  Sharpe Ratio  : %.2f", bh_sharpe)
        logger.info("-----------------------------------------------------")

        test_df.to_csv(config.BACKTEST_RESULTS_PATH)
        logger.info("Detailed trade log exported to %s", config.BACKTEST_RESULTS_PATH)

        return test_df

    except Exception as e:
        logger.error("An error occurred during backtesting: %s", str(e), exc_info=True)
        return None
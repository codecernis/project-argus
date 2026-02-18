"""
Project Argus Controller.
Orchestrates the quantitative pipeline: Data Ingestion -> Feature Engineering -> ML Training -> Backtesting.
"""

import logging
import sys
from src.data_ingestion import fetch_market_data
from src.features import generate_features
from src.model import train_and_predict_wfo
from src.backtest import run_backtest

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main() -> None:
    """Main execution function for the Project Argus pipeline."""
    logger.info("========================================")
    logger.info("   PROJECT ARGUS: INITIALIZING SYSTEM   ")
    logger.info("========================================")

    # Phase 1: Data Extraction
    raw_df = fetch_market_data()
    if raw_df is None:
        logger.critical("Pipeline halted: Data Ingestion failed.")
        sys.exit(1)

    # Phase 2: Alpha Generation (Feature Engineering)
    processed_df = generate_features(raw_df)
    if processed_df is None or processed_df.empty:
        logger.critical("Pipeline halted: Feature Engineering failed.")
        sys.exit(1)

    # Phase 3: Walk-Forward Optimization (ML Training)
    wfo_df = train_and_predict_wfo(processed_df)
    if wfo_df is None or wfo_df.empty:
        logger.critical("Pipeline halted: Model Training failed.")
        sys.exit(1)

    # Phase 4: Historical Simulation & Risk Analysis
    results_df = run_backtest(wfo_df)
    if results_df is None:
        logger.critical("Pipeline halted: Backtesting failed.")
        sys.exit(1)

    logger.info("========================================")
    logger.info("   PROJECT ARGUS: PIPELINE COMPLETE     ")
    logger.info("========================================")

if __name__ == "__main__":
    main()
"""
Configuration module for Project Argus.
Stores all hyper-parameters, file paths, and system constants.
"""

from pathlib import Path

# --- Data Settings ---
SYMBOL: str = "BTC-USD"
PERIOD: str = "max"
INTERVAL: str = "1d"

# --- Feature Engineering Settings ---
RSI_LENGTH: int = 14
EMA_LENGTH: int = 20
BB_LENGTH: int = 20
LOOKAHEAD_PERIODS: int = 5

# --- Machine Learning Settings ---
N_ESTIMATORS: int = 100
RANDOM_STATE: int = 42
TRAIN_WINDOW_DAYS: int = 365
STEP_SIZE_DAYS: int = 30

# --- Backtest Settings ---
STARTING_BALANCE: float = 10000.00
TRADING_FEE: float = 0.001
TRADING_DAYS_PER_YEAR: int = 365

# --- File Paths ---
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"

RAW_DATA_PATH: Path = DATA_DIR / "raw_data.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "argus_market_data.csv"
BACKTEST_RESULTS_PATH: Path = DATA_DIR / "backtest_results.csv"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
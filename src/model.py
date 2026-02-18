"""
Machine Learning Module.
Implements Walk-Forward Optimization (WFO) using a Random Forest Classifier
to account for market regime changes over time.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src import config

logger = logging.getLogger(__name__)

def train_and_predict_wfo(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Trains a series of rolling machine learning models using Walk-Forward Optimization.
    Predicts out-of-sample data sequentially to prevent data leakage and adapt to regime shifts.

    Args:
        df (pd.DataFrame): Processed data containing features and target labels.

    Returns:
        pd.DataFrame: The original dataframe appended with the continuous out-of-sample predictions.
    """
    logger.info("Initializing Walk-Forward Optimization (WFO) training engine...")

    feature_cols = [
        f"RSI_{config.RSI_LENGTH}", 
        f"EMA_{config.EMA_LENGTH}", 
        f"BBP_{config.BB_LENGTH}_2.0",
        "MACDh_12_26_9",
        "ATRr_14"
    ]

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing expected feature columns: %s", missing_cols)
        return None

    X = df[feature_cols]
    y = df['Target']

    predictions = pd.Series(index=df.index, dtype=float)
    feature_importances = np.zeros(len(feature_cols))

    total_steps = (len(df) - config.TRAIN_WINDOW_DAYS) // config.STEP_SIZE_DAYS
    logger.info(
        "WFO Configuration: %d-day training window, %d-day step size. Total models to train: ~%d.",
        config.TRAIN_WINDOW_DAYS, 
        config.STEP_SIZE_DAYS, 
        total_steps
    )

    models_trained = 0

    try:
        for i in range(config.TRAIN_WINDOW_DAYS, len(df), config.STEP_SIZE_DAYS):
            # Define In-Sample (Training) Window
            start_idx = i - config.TRAIN_WINDOW_DAYS
            X_train = X.iloc[start_idx:i]
            y_train = y.iloc[start_idx:i]

            # Define Out-of-Sample (Testing) Window
            end_idx = min(i + config.STEP_SIZE_DAYS, len(df))
            X_test = X.iloc[i:end_idx]

            # Utilize all available CPU cores (n_jobs=-1) for parallel tree building
            model = RandomForestClassifier(
                n_estimators=config.N_ESTIMATORS, 
                random_state=config.RANDOM_STATE,
                n_jobs=-1  
            )
            model.fit(X_train, y_train)

            predictions.iloc[i:end_idx] = model.predict(X_test)
            feature_importances += model.feature_importances_
            models_trained += 1

        df_result = df.copy()
        df_result['Prediction'] = predictions
        
        # Discard the initial training window as it lacks predictions
        df_result = df_result.iloc[config.TRAIN_WINDOW_DAYS:].copy()

        if models_trained > 0:
            avg_importances = feature_importances / models_trained
            logger.info("Average Feature Importance (Across %d models):", models_trained)
            feature_scores = sorted(zip(feature_cols, avg_importances), key=lambda x: x[1], reverse=True)
            for feature, score in feature_scores:
                logger.info("  %s: %.2f%%", feature, score * 100)

        return df_result

    except Exception as e:
        logger.error("An error occurred during WFO training: %s", str(e), exc_info=True)
        return None
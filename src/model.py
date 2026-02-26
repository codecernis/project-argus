"""
Phase 6 Fallback — RF-only Walk-Forward Optimization.

Used when LightGBM is unavailable or ensemble is disabled.

Walk-forward structure per fold:
  [====== TRAIN (365d) ======]---PURGE (10d)---[EMBARGO (5d)|==== TEST (30d) ====]

Per-fold outputs appended as DataFrame columns:
  Probability                    — RF P(win) for test bars
  empirical_avg_class0_return    — mean pnl_pct of label-0 training samples (Kelly loss)
  empirical_avg_class1_return    — mean pnl_pct of label-1 training samples (Kelly gain)
  fold_id                        — integer fold index for auditability

Per-fold logging:
  n_train_samples, n_test_samples, samples_to_features_ratio,
  feature importances (top-10), Kelly calibration data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import ArgusConfig
from src.exceptions import ModelError
from src.features import get_feature_columns

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data container for per-fold results (used by backtest reporting)
# -------------------------------------------------------------------

@dataclass
class FoldMetadata:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_features: int
    feature_importances: dict[str, float] = field(default_factory=dict)
    empirical_avg_class0_return: float = 0.0
    empirical_avg_class1_return: float = 0.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _time_decay_weights(n: int, lam: float) -> np.ndarray:
    """Exponential time-decay: most recent sample gets weight 1, oldest gets exp(-lam)."""
    if lam == 0.0:
        return np.ones(n)
    t = np.linspace(0, 1, n)
    w = np.exp(lam * (t - 1.0))   # w[−1] = 1, w[0] = exp(-lam)
    return w / w.sum() * n        # keep effective sample count comparable


def _log_feature_importances(
    feature_cols: list[str], importances: np.ndarray, top_n: int = 10
) -> dict[str, float]:
    pairs = sorted(
        zip(feature_cols, importances), key=lambda x: x[1], reverse=True
    )
    top = pairs[:top_n]
    logger.debug("    Top-%d feature importances:", top_n)
    for fname, score in top:
        logger.debug("      %-30s %.4f", fname, score)
    return {fname: float(score) for fname, score in pairs}


# -------------------------------------------------------------------
# Public WFO function
# -------------------------------------------------------------------

def train_and_predict_wfo(
    df: pd.DataFrame, cfg: ArgusConfig
) -> tuple[pd.DataFrame, list[FoldMetadata]]:
    """
    Purged walk-forward optimisation using a Random Forest classifier.

    Returns
    -------
    df_out : pd.DataFrame
        Original DataFrame with Probability, empirical_avg_class0_return,
        empirical_avg_class1_return, and fold_id columns added for all
        bars that fall within a test window.
    fold_metadata : list[FoldMetadata]
        Per-fold diagnostics for backtest reporting.
    """
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ModelError("No feature columns found in DataFrame. Run feature engineering first.")

    required = ["Target", "pnl_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ModelError(f"Missing required columns for WFO: {missing}")

    logger.info(
        "Phase 6 (RF-WFO) — %d features | train=%dd | step=%dd | purge=%dd | embargo=%dd",
        len(feature_cols),
        cfg.model.train_window_days,
        cfg.model.step_size_days,
        cfg.model.purge_days,
        cfg.model.embargo_days,
    )

    n = len(df)
    gap = cfg.model.purge_days + cfg.model.embargo_days

    # Output arrays (NaN for bars not covered by any test window)
    prob_out = np.full(n, np.nan)
    class0_return_out = np.full(n, np.nan)
    class1_return_out = np.full(n, np.nan)
    fold_id_out = np.full(n, -1, dtype=int)

    fold_metadata: list[FoldMetadata] = []
    fold_id = 0

    # WFO slide: test_start advances by step_size_days each fold
    test_start = cfg.model.train_window_days + gap

    try:
        while test_start < n:
            test_end = min(test_start + cfg.model.step_size_days, n)
            train_end = test_start - gap
            train_start = max(0, train_end - cfg.model.train_window_days)

            if train_end - train_start < cfg.model.step_size_days:
                break  # Not enough training data

            X_train = df[feature_cols].iloc[train_start:train_end].values
            y_train = df["Target"].iloc[train_start:train_end].values
            pnl_train = df["pnl_pct"].iloc[train_start:train_end].values

            n_train = len(X_train)
            n_test = test_end - test_start
            ratio = n_train / len(feature_cols)

            logger.info(
                "  Fold %d | train [%s → %s] | test [%s → %s]",
                fold_id,
                df.index[train_start].date(),
                df.index[train_end - 1].date(),
                df.index[test_start].date(),
                df.index[test_end - 1].date(),
            )
            logger.debug(
                "    n_train=%d | n_test=%d | features=%d | ratio=%.1f:1",
                n_train,
                n_test,
                len(feature_cols),
                ratio,
            )

            if ratio < cfg.model.min_samples_to_features_ratio:
                logger.warning(
                    "    Fold %d: samples-to-features ratio %.1f:1 < %.0f:1 threshold. "
                    "Consider wider train window.",
                    fold_id,
                    ratio,
                    cfg.model.min_samples_to_features_ratio,
                )

            # Time-decay sample weights
            weights = _time_decay_weights(n_train, cfg.model.time_decay_lambda)

            # Kelly calibration data (empirical returns from training fold)
            mask0 = y_train == 0
            mask1 = y_train == 1
            avg_class0 = float(pnl_train[mask0].mean()) if mask0.any() else -0.01
            avg_class1 = float(pnl_train[mask1].mean()) if mask1.any() else 0.01

            rf = RandomForestClassifier(
                n_estimators=cfg.model.n_estimators,
                max_depth=cfg.model.max_depth,
                min_samples_leaf=cfg.model.min_samples_leaf,
                random_state=cfg.model.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train, sample_weight=weights)

            X_test = df[feature_cols].iloc[test_start:test_end].values
            probs = rf.predict_proba(X_test)

            # Identify which column corresponds to class 1
            class1_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else -1
            if class1_idx == -1:
                p_win = np.full(n_test, 0.5)
            else:
                p_win = probs[:, class1_idx]

            prob_out[test_start:test_end] = p_win
            class0_return_out[test_start:test_end] = avg_class0
            class1_return_out[test_start:test_end] = avg_class1
            fold_id_out[test_start:test_end] = fold_id

            importances = _log_feature_importances(feature_cols, rf.feature_importances_)

            fold_metadata.append(
                FoldMetadata(
                    fold_id=fold_id,
                    train_start=str(df.index[train_start].date()),
                    train_end=str(df.index[train_end - 1].date()),
                    test_start=str(df.index[test_start].date()),
                    test_end=str(df.index[test_end - 1].date()),
                    n_train=n_train,
                    n_test=n_test,
                    n_features=len(feature_cols),
                    feature_importances=importances,
                    empirical_avg_class0_return=avg_class0,
                    empirical_avg_class1_return=avg_class1,
                )
            )

            fold_id += 1
            test_start += cfg.model.step_size_days

    except ModelError:
        raise
    except Exception as exc:
        raise ModelError(f"WFO training failed at fold {fold_id}: {exc}") from exc

    if fold_id == 0:
        raise ModelError(
            "WFO produced zero folds. Dataset may be too small for the configured "
            f"train_window_days={cfg.model.train_window_days}."
        )

    logger.info(
        "Phase 6 (RF-WFO) complete — %d folds trained, %d bars covered.",
        fold_id,
        int(np.sum(~np.isnan(prob_out))),
    )

    df_out = df.copy()
    df_out["Probability"] = prob_out
    df_out["empirical_avg_class0_return"] = class0_return_out
    df_out["empirical_avg_class1_return"] = class1_return_out
    df_out["fold_id"] = fold_id_out

    return df_out, fold_metadata

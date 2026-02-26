"""
Phase 6 — RF + LightGBM Ensemble with Scalar-Optimised Blender.

Walk-forward structure mirrors model.py with the addition of a meta-learner:

  RF  → P(win|RF)  ─┐
                     ├─→ Scalar-Optimised Blender → Blended P(win)
  LGBM→ P(win|LGBM)─┘

Scalar-Optimised Blender (v3.3):
  Since w2 = 1 − w1, this is a 1-D problem:
    w1* = argmin  log_loss(y, w1 × P_rf + (1 − w1) × P_lgbm)
           w1∈[0,1]
  Solved via scipy.optimize.minimize_scalar(method='bounded').

Burn-in rule:
  Folds 1 .. burn_in:  w1 = 0.5 (equal weights)
  Folds burn_in+1 .. N: blender trained on previous fold's test data.

Falls back to model.py (RF-only WFO) if LightGBM is unavailable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar  # type: ignore
from sklearn.metrics import log_loss

from src.config import ArgusConfig
from src.exceptions import EnsembleError
from src.features import get_feature_columns
from src.model import FoldMetadata, _log_feature_importances, _time_decay_weights

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb  # type: ignore

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# -------------------------------------------------------------------
# Scalar blender optimisation
# -------------------------------------------------------------------

def _find_optimal_weight(
    p_rf: np.ndarray, p_lgbm: np.ndarray, y_true: np.ndarray
) -> float:
    """
    Find w1* that minimises log-loss(y, w1*P_rf + (1-w1)*P_lgbm) over [0, 1].

    Returns w1 (weight on RF).  Sub-millisecond; guaranteed convergence.
    """
    def objective(w: float) -> float:
        p_blend = np.clip(w * p_rf + (1.0 - w) * p_lgbm, 1e-9, 1.0 - 1e-9)
        return float(log_loss(y_true, p_blend, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def train_and_predict_ensemble(
    df: pd.DataFrame, cfg: ArgusConfig
) -> tuple[pd.DataFrame, list[FoldMetadata]]:
    """
    Ensemble WFO: RF + LightGBM with scalar-optimised blending.

    Falls back to RF-only (model.py) if LightGBM is unavailable.

    Returns
    -------
    df_out : pd.DataFrame
        Input DataFrame with Probability, empirical_avg_class{0,1}_return,
        and fold_id columns added.
    fold_metadata : list[FoldMetadata]
    """
    if not cfg.ensemble.enabled:
        logger.info("Phase 6 — Ensemble disabled; falling back to RF-only WFO.")
        from src.model import train_and_predict_wfo
        return train_and_predict_wfo(df, cfg)

    if not LGBM_AVAILABLE:
        logger.warning(
            "Phase 6 — LightGBM not installed; falling back to RF-only WFO. "
            "Install lightgbm for the full ensemble."
        )
        from src.model import train_and_predict_wfo
        return train_and_predict_wfo(df, cfg)

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise EnsembleError(
            "No feature columns found in DataFrame. Run feature engineering first."
        )

    required = ["Target", "pnl_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise EnsembleError(f"Missing required columns: {missing}")

    logger.info(
        "Phase 6 (Ensemble) — %d features | burn_in=%d | train=%dd | step=%dd",
        len(feature_cols),
        cfg.ensemble.meta_learner_burn_in,
        cfg.model.train_window_days,
        cfg.model.step_size_days,
    )

    n = len(df)
    gap = cfg.model.purge_days + cfg.model.embargo_days

    prob_out = np.full(n, np.nan)
    class0_return_out = np.full(n, np.nan)
    class1_return_out = np.full(n, np.nan)
    fold_id_out = np.full(n, -1, dtype=int)

    fold_metadata: list[FoldMetadata] = []
    fold_id = 0

    # State carried across folds for the blender
    prev_p_rf: np.ndarray | None = None
    prev_p_lgbm: np.ndarray | None = None
    prev_y: np.ndarray | None = None
    w1: float = 0.5  # default equal-weight (burn-in)

    test_start = cfg.model.train_window_days + gap

    try:
        while test_start < n:
            test_end = min(test_start + cfg.model.step_size_days, n)
            train_end = test_start - gap
            train_start = max(0, train_end - cfg.model.train_window_days)

            if train_end - train_start < cfg.model.step_size_days:
                break

            X_train = df[feature_cols].iloc[train_start:train_end].values
            y_train = df["Target"].iloc[train_start:train_end].values
            pnl_train = df["pnl_pct"].iloc[train_start:train_end].values
            X_test = df[feature_cols].iloc[test_start:test_end].values
            y_test = df["Target"].iloc[test_start:test_end].values

            n_train = len(X_train)
            n_test = test_end - test_start
            ratio = n_train / len(feature_cols)

            logger.info(
                "  Fold %d | train [%s → %s] | test [%s → %s] | n=%d | ratio=%.1f:1",
                fold_id,
                df.index[train_start].date(),
                df.index[train_end - 1].date(),
                df.index[test_start].date(),
                df.index[test_end - 1].date(),
                n_train,
                ratio,
            )

            if ratio < cfg.model.min_samples_to_features_ratio:
                logger.warning(
                    "    Fold %d: samples-to-features ratio %.1f:1 < %.0f:1.",
                    fold_id,
                    ratio,
                    cfg.model.min_samples_to_features_ratio,
                )

            weights = _time_decay_weights(n_train, cfg.model.time_decay_lambda)

            # Kelly calibration
            mask0 = y_train == 0
            mask1 = y_train == 1
            avg_class0 = float(pnl_train[mask0].mean()) if mask0.any() else -0.01
            avg_class1 = float(pnl_train[mask1].mean()) if mask1.any() else 0.01

            # --- Train RF ---
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(
                n_estimators=cfg.model.n_estimators,
                max_depth=cfg.model.max_depth,
                min_samples_leaf=cfg.model.min_samples_leaf,
                random_state=cfg.model.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train, sample_weight=weights)

            c1_rf = list(rf.classes_).index(1) if 1 in rf.classes_ else -1
            p_rf = (
                rf.predict_proba(X_test)[:, c1_rf]
                if c1_rf != -1
                else np.full(n_test, 0.5)
            )

            # --- Train LightGBM ---
            lgb_params = {
                "objective": "binary",
                "n_estimators": cfg.ensemble.lgbm_n_estimators,
                "learning_rate": cfg.ensemble.lgbm_learning_rate,
                "max_depth": cfg.ensemble.lgbm_max_depth,
                "num_leaves": cfg.ensemble.lgbm_num_leaves,
                "random_state": cfg.model.random_state,
                "verbose": -1,
            }
            gbm = lgb.LGBMClassifier(**lgb_params)
            gbm.fit(X_train, y_train, sample_weight=weights)

            c1_lgbm = list(gbm.classes_).index(1) if 1 in gbm.classes_ else -1
            p_lgbm = (
                gbm.predict_proba(X_test)[:, c1_lgbm]
                if c1_lgbm != -1
                else np.full(n_test, 0.5)
            )

            # --- Scalar blender ---
            burn_in = cfg.ensemble.meta_learner_burn_in
            fold_number = fold_id + 1  # 1-indexed per blueprint

            if fold_number <= burn_in:
                w1 = 0.5
                logger.debug("    Fold %d (burn-in): w1=0.50 (equal weights).", fold_id)
            elif prev_p_rf is not None and prev_y is not None and len(np.unique(prev_y)) >= 2:
                w1 = _find_optimal_weight(prev_p_rf, prev_p_lgbm, prev_y)
                logger.debug("    Fold %d: blender w1=%.4f.", fold_id, w1)
            else:
                w1 = 0.5
                if prev_y is not None and len(np.unique(prev_y)) < 2:
                    logger.debug(
                        "    Fold %d: single-class calibration fold — using w1=0.50.",
                        fold_id,
                    )

            p_blend = w1 * p_rf + (1.0 - w1) * p_lgbm

            prob_out[test_start:test_end] = p_blend
            class0_return_out[test_start:test_end] = avg_class0
            class1_return_out[test_start:test_end] = avg_class1
            fold_id_out[test_start:test_end] = fold_id

            # Save current fold's test predictions for next fold's blender training
            prev_p_rf = p_rf
            prev_p_lgbm = p_lgbm
            prev_y = y_test

            # Feature importances (RF + LightGBM)
            rf_imp = rf.feature_importances_
            lgb_imp_raw = gbm.feature_importances_
            lgb_imp = lgb_imp_raw / (lgb_imp_raw.sum() or 1.0)
            combined_imp = w1 * rf_imp + (1.0 - w1) * lgb_imp
            importances = _log_feature_importances(feature_cols, combined_imp)

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

    except EnsembleError:
        raise
    except Exception as exc:
        raise EnsembleError(
            f"Ensemble WFO failed at fold {fold_id}: {exc}"
        ) from exc

    if fold_id == 0:
        raise EnsembleError(
            "Ensemble WFO produced zero folds. Dataset may be too small."
        )

    logger.info(
        "Phase 6 (Ensemble) complete — %d folds | %d bars covered.",
        fold_id,
        int(np.sum(~np.isnan(prob_out))),
    )

    df_out = df.copy()
    df_out["Probability"] = prob_out
    df_out["empirical_avg_class0_return"] = class0_return_out
    df_out["empirical_avg_class1_return"] = class1_return_out
    df_out["fold_id"] = fold_id_out

    return df_out, fold_metadata

"""
Phase 4 — HMM Regime Detection.

Trains a Gaussian HMM on (returns, realised_vol_20d, volume_pct_change) using a
ROLLING window (not expanding) to prevent anchoring to early BTC history.

Walk-forward: for each bar the model is trained on the most recent
`train_window_days` bars and predicts ONLY the current bar (no look-ahead).

States are auto-labelled by mean return:
  highest mean → bull, lowest mean → bear, middle → neutral.

Output columns added: Regime, Regime_Confidence, Regime_Is_Bear.

If hmmlearn is unavailable, all bars are labelled neutral (Regime=1,
Regime_Confidence=1.0, Regime_Is_Bear=0) with a logged warning.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import RegimeError

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm as _hmm_module  # type: ignore

    HMM_AVAILABLE = True
    logging.getLogger("hmmlearn.base").setLevel(logging.CRITICAL)
except ImportError:
    HMM_AVAILABLE = False


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _build_obs_matrix(df_window: pd.DataFrame) -> np.ndarray | None:
    """Stack (returns, realised_vol_20d, volume_pct_change) into an (n, 3) array."""
    returns = df_window["Close"].pct_change()
    vol = df_window["realised_vol_20d"]
    vol_chg = df_window["Volume"].pct_change()

    obs = pd.DataFrame(
        {
            "ret": returns,
            "vol": vol,
            "vol_chg": vol_chg,
        }
    ).dropna()

    if len(obs) < 10:
        return None
    return obs.values


def _label_states_by_mean_return(
    model: object, obs: np.ndarray, n_regimes: int
) -> dict[int, str]:
    """
    Assign semantic labels to HMM states by sorting on mean return of training data.

    Returns a dict: {hmm_state_id: "bull" | "neutral" | "bear"}.
    """
    # Predict states on training data to compute per-state mean return
    state_seq = model.predict(obs)  # type: ignore[attr-defined]
    returns = obs[:, 0]

    state_means: dict[int, float] = {}
    for s in range(n_regimes):
        mask = state_seq == s
        if mask.any():
            state_means[s] = float(returns[mask].mean())
        else:
            state_means[s] = 0.0  # no samples assigned to this state

    # Filter out NaN means (degenerate fits) before sorting
    state_means = {s: (v if np.isfinite(v) else 0.0) for s, v in state_means.items()}
    sorted_states = sorted(state_means, key=lambda s: state_means[s])  # ascending by mean

    labels: dict[int, str] = {}
    bear_mean = state_means[sorted_states[0]]
    next_mean = state_means[sorted_states[1]]
    # Bear requires negative mean AND meaningful separation (> 50 bps daily)
    is_bear = bear_mean < 0 and (next_mean - bear_mean) > 0.005

    if n_regimes == 2:
        labels[sorted_states[0]] = "bear" if is_bear else "neutral"
        labels[sorted_states[1]] = "bull"
    else:
        labels[sorted_states[0]] = "bear" if is_bear else "neutral"
        labels[sorted_states[-1]] = "bull"
        for s in sorted_states[1:-1]:
            labels[s] = "neutral"

    return labels


def _state_to_regime_id(label: str) -> int:
    return {"bear": 0, "neutral": 1, "bull": 2}.get(label, 1)


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def add_regime_features(df: pd.DataFrame, cfg: ArgusConfig) -> pd.DataFrame:
    """
    Add Regime, Regime_Confidence, Regime_Is_Bear columns to *df*.

    Raises RegimeError on any unrecoverable failure.
    """
    if not cfg.regime.enabled:
        logger.info("Phase 4 — Regime detection disabled; assigning neutral regime.")
        result = df.copy()
        result["Regime"] = 1
        result["Regime_Confidence"] = 1.0
        result["Regime_Is_Bear"] = 0
        return result

    if not HMM_AVAILABLE:
        logger.warning(
            "Phase 4 — hmmlearn not installed. Assigning neutral regime to all bars. "
            "Install hmmlearn for live regime detection."
        )
        result = df.copy()
        result["Regime"] = 1
        result["Regime_Confidence"] = 1.0
        result["Regime_Is_Bear"] = 0
        return result

    if "realised_vol_20d" not in df.columns:
        raise RegimeError(
            "Column 'realised_vol_20d' not found. Run features.generate_features() first."
        )

    logger.info(
        "Phase 4 — Walk-forward HMM regime detection | n_regimes=%d | window=%s.",
        cfg.regime.n_regimes,
        cfg.regime.window_type,
    )

    try:
        n = len(df)
        window = cfg.model.train_window_days
        n_regimes = cfg.regime.n_regimes
        refit_every = cfg.model.step_size_days  # default 30

        regime_ids = np.full(n, 1, dtype=int)        # default: neutral
        confidences = np.zeros(n, dtype=float)        # 0.0 = no HMM fit yet
        is_bear = np.zeros(n, dtype=int)

        # Counters
        _hmm_attempted = 0
        _hmm_converged = 0
        _hmm_failed_count = 0

        # Cached values from last successful refit (start neutral)
        cached_regime = 1
        cached_confidence = 0.0
        cached_is_bear = 0
        last_refit_bar = -refit_every  # force refit on first eligible bar

        # Precompute arrays for bar-level prediction
        close_arr = df["Close"].values
        vol_arr = df["realised_vol_20d"].values
        vol_pct_chg = df["Volume"].pct_change().values
        conf_threshold = cfg.regime.regime_confidence_threshold

        for i in range(n):
            # --- Carry forward on non-refit bars ---
            if (i - last_refit_bar) < refit_every:
                regime_ids[i] = cached_regime
                confidences[i] = cached_confidence
                is_bear[i] = cached_is_bear
                if (i + 1) % 500 == 0:
                    logger.info(
                        "HMM progress: %d / %d bars (%.0f%%)",
                        i + 1, n, (i + 1) / n * 100,
                    )
                continue

            # --- Refit bar: entire block wrapped in try/except ---
            try:
                if cfg.regime.window_type == "rolling":
                    start = max(0, i - window)
                else:
                    start = 0

                df_window = df.iloc[start:i]
                obs = _build_obs_matrix(df_window)

                if obs is None or len(obs) < n_regimes * 10:
                    # Not enough data — carry forward, don't count as attempt
                    regime_ids[i] = cached_regime
                    confidences[i] = cached_confidence
                    is_bear[i] = cached_is_bear
                    if (i + 1) % 500 == 0:
                        logger.info(
                            "HMM progress: %d / %d bars (%.0f%%)",
                            i + 1, n, (i + 1) / n * 100,
                        )
                    continue

                # Try 3 random seeds, keep best converged model
                _hmm_attempted += 1
                best_model = None
                best_score = -np.inf

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for seed in (42, 43, 44):
                        candidate = _hmm_module.GaussianHMM(
                            n_components=n_regimes,
                            covariance_type=cfg.regime.covariance_type,
                            n_iter=cfg.regime.n_iter,
                            random_state=seed,
                        )
                        try:
                            candidate.fit(obs)
                            if candidate.monitor_.converged:
                                score = candidate.score(obs)
                                if score > best_score:
                                    best_score = score
                                    best_model = candidate
                        except Exception:
                            continue

                if best_model is None:
                    # None of the 3 candidates converged — neutral fallback
                    _hmm_failed_count += 1
                    cached_regime = 1
                    cached_confidence = 0.0
                    cached_is_bear = 0
                    regime_ids[i] = cached_regime
                    confidences[i] = cached_confidence
                    is_bear[i] = cached_is_bear
                    last_refit_bar = i
                    if (i + 1) % 500 == 0:
                        logger.info(
                            "HMM progress: %d / %d bars (%.0f%%)",
                            i + 1, n, (i + 1) / n * 100,
                        )
                    continue

                _hmm_converged += 1
                model = best_model
                last_refit_bar = i

                # Predict current bar
                current_ret = close_arr[i] / close_arr[max(0, i - 1)] - 1.0
                current_vol = vol_arr[i]
                current_vol_chg = vol_pct_chg[i]

                if any(np.isnan([current_ret, current_vol, current_vol_chg])):
                    cached_regime = 1
                    cached_confidence = 0.0
                    cached_is_bear = 0
                    regime_ids[i] = cached_regime
                    confidences[i] = cached_confidence
                    is_bear[i] = cached_is_bear
                    if (i + 1) % 500 == 0:
                        logger.info(
                            "HMM progress: %d / %d bars (%.0f%%)",
                            i + 1, n, (i + 1) / n * 100,
                        )
                    continue

                bar_obs = np.array([[current_ret, current_vol, current_vol_chg]])
                state_probs = model.predict_proba(bar_obs)[0]
                predicted_state = int(np.argmax(state_probs))
                confidence = float(state_probs[predicted_state])
                state_labels = _label_states_by_mean_return(model, obs, n_regimes)
                label = state_labels.get(predicted_state, "neutral")

                # Bear gating: only assign bear when confidence exceeds threshold
                if label == "bear" and confidence < conf_threshold:
                    label = "neutral"

                cached_regime = _state_to_regime_id(label)
                cached_confidence = confidence
                cached_is_bear = 1 if label == "bear" else 0

                regime_ids[i] = cached_regime
                confidences[i] = cached_confidence
                is_bear[i] = cached_is_bear

            except Exception:
                # Any per-bar failure → neutral fallback
                _hmm_failed_count += 1
                cached_regime = 1
                cached_confidence = 0.0
                cached_is_bear = 0
                regime_ids[i] = cached_regime
                confidences[i] = cached_confidence
                is_bear[i] = cached_is_bear

            if (i + 1) % 500 == 0:
                logger.info(
                    "HMM progress: %d / %d bars (%.0f%%)",
                    i + 1, n, (i + 1) / n * 100,
                )

        # --- One summary line ---
        bear_count = int(is_bear.sum())
        bear_pct = bear_count / n * 100 if n > 0 else 0.0
        conv_pct = (
            _hmm_converged / _hmm_attempted * 100 if _hmm_attempted > 0 else 0.0
        )
        logger.info(
            "Phase 4 — HMM summary | windows=%d | converged=%d (%.0f%%) "
            "| fell_back_to_neutral=%d | bear_bars=%d (%.1f%%)",
            _hmm_attempted,
            _hmm_converged,
            conv_pct,
            _hmm_failed_count,
            bear_count,
            bear_pct,
        )

        if _hmm_attempted > 0 and conv_pct < 50.0:
            logger.warning(
                "HMM convergence rate %.0f%% < 50%%. Consider using n_regimes=2.",
                conv_pct,
            )

        result = df.copy()
        result["Regime"] = regime_ids
        result["Regime_Confidence"] = confidences
        result["Regime_Is_Bear"] = is_bear
        return result

    except RegimeError:
        raise
    except Exception as exc:
        raise RegimeError(f"Regime detection failed: {exc}") from exc

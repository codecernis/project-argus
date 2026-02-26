"""
Phase 7 — Dynamic Kelly Position Sizing with 4 Layers + Circuit Breaker + Hysteresis.

Position sizing layers (in order):
  1. Kelly (dynamic, per-bar):
       f_full = p − q/b   where b = pt_distance / sl_distance (today's barriers)
       f = f_full × kelly_fraction   (default fractional Kelly = 0.40)
       Guard: if f_full ≤ 0 or sl_distance ≈ 0 → f = 0

  2. VolTgt scalar:
       s = vol_target / realised_vol_20d   (capped at 2×)

  3. DD multiplier:
       m = 1 + drawdown / |max_drawdown_halt|   (linear 1→0 as DD→max)

  4. Regime multiplier:
       r = 0.0 if bear + confident HMM state
       r = regime_neutral_multiplier if neutral HMM state
       r = 1.0 if bull

  raw_target = f × s × m × r   (clipped to [0, 1])

  5. Circuit-Breaker Bypass (BEFORE hysteresis):
       if r == 0 → Position_Size = 0, skip hysteresis, log event
       if m ≤ 0  → Position_Size = 0, skip hysteresis, log event

  6. Hysteresis Gate:
       if |raw_target − current_position| < rebalance_threshold → NO TRADE
       else → rebalance to raw_target

Output columns added: Position_Size, circuit_breaker_fired, hysteresis_blocked.

Also returns a summary dict with hysteresis / circuit-breaker statistics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import RiskError

logger = logging.getLogger(__name__)


@dataclass
class RiskStats:
    total_rebalance_events: int = 0
    suppressed_by_hysteresis: int = 0
    circuit_breaker_activations: int = 0
    # Per-layer kill counters (Fix 4: diagnostic visibility)
    killed_by_kelly: int = 0       # f_full ≤ 0 (no edge)
    killed_by_regime_cb: int = 0   # r == 0 (bear + confident)
    killed_by_dd_cb: int = 0       # m ≤ 0 (drawdown halt)
    re_entries_after_flatten: int = 0

    @property
    def suppression_rate(self) -> float:
        total = self.total_rebalance_events + self.suppressed_by_hysteresis
        return self.suppressed_by_hysteresis / total if total > 0 else 0.0


# -------------------------------------------------------------------
# Per-bar Kelly sizing (pure function for testability)
# -------------------------------------------------------------------

def _kelly_fraction(
    p: float,
    pt_dist: float,
    sl_dist: float,
    avg_class1_return: float,
    avg_class0_return: float,
    cfg: ArgusConfig,
) -> float:
    """
    Compute fractional Kelly for one bar.

    Returns f ∈ [0, 1].  Returns 0 if there is no mathematical edge.
    """
    q = 1.0 - p

    if cfg.risk.kelly_use_dynamic_odds:
        if sl_dist < cfg.risk.min_sl_distance:
            return 0.0  # undefined odds → no trade
        b = pt_dist / sl_dist
    else:
        # Static per-fold fallback using empirical avg returns
        loss_est = abs(avg_class0_return) if avg_class0_return != 0 else cfg.risk.min_sl_distance
        b = avg_class1_return / loss_est if loss_est > 0 else 0.0

    if b <= 0:
        return 0.0

    f_full = p - q / b      # Kelly criterion — trust the math

    if f_full <= 0:
        return 0.0           # No edge → no trade (no heuristic p/b guards per v3.3)

    return float(np.clip(f_full * cfg.risk.kelly_fraction, 0.0, 1.0))


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def compute_positions(df: pd.DataFrame, cfg: ArgusConfig) -> tuple[pd.DataFrame, RiskStats]:
    """
    Compute Position_Size for every bar via the 4-layer + circuit-breaker + hysteresis model.

    Requires columns: Probability, pt_distance, sl_distance,
                      empirical_avg_class1_return, empirical_avg_class0_return,
                      realised_vol_20d, Regime_Confidence, Regime_Is_Bear, Close.

    Iterates bar-by-bar to maintain state (current position, equity, peak equity).

    Returns (df_out, RiskStats).  Raises RiskError on failure.
    """
    required = [
        "Probability",
        "pt_distance",
        "sl_distance",
        "empirical_avg_class1_return",
        "empirical_avg_class0_return",
        "realised_vol_20d",
        "Close",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RiskError(
            f"Missing required columns for risk sizing: {missing}. "
            "Ensure model WFO and feature engineering have run."
        )

    n = len(df)
    position_out = np.zeros(n)
    cb_fired_out = np.zeros(n, dtype=bool)
    hysteresis_out = np.zeros(n, dtype=bool)

    # Internal state
    current_position: float = 0.0
    equity: float = cfg.backtest.starting_balance
    peak_equity: float = equity
    flat_fee = (cfg.backtest.trading_fee_bps + cfg.backtest.slippage_bps) / 10_000.0

    stats = RiskStats()

    close_arr = df["Close"].values
    prob_arr = df["Probability"].values
    pt_arr = df["pt_distance"].values
    sl_arr = df["sl_distance"].values
    c1_arr = df["empirical_avg_class1_return"].values
    c0_arr = df["empirical_avg_class0_return"].values
    vol_arr = df["realised_vol_20d"].values

    regime_is_bear = df.get("Regime_Is_Bear", pd.Series(np.zeros(n))).values
    regime_conf = df.get(
        "Regime_Confidence", pd.Series(np.ones(n))
    ).values
    regime_id_arr = df.get("Regime", pd.Series(np.ones(n, dtype=int))).values

    try:
        for i in range(n):
            # --- Update equity with previous bar's return ---
            if i > 0 and close_arr[i - 1] > 0:
                bar_ret = (close_arr[i] - close_arr[i - 1]) / close_arr[i - 1]
                equity *= 1.0 + current_position * bar_ret
                peak_equity = max(peak_equity, equity)

            p = float(prob_arr[i])
            if np.isnan(p):
                # Bar not covered by a test window — hold current position
                position_out[i] = current_position
                continue

            pt_d = float(pt_arr[i])
            sl_d = float(sl_arr[i])
            avg_c1 = float(c1_arr[i]) if not np.isnan(c1_arr[i]) else 0.01
            avg_c0 = float(c0_arr[i]) if not np.isnan(c0_arr[i]) else -0.01
            vol = float(vol_arr[i])

            # --- Layer 1: Kelly ---
            f = _kelly_fraction(p, pt_d, sl_d, avg_c1, avg_c0, cfg)

            # --- Layer 2: VolTgt scalar ---
            if vol > 0:
                s = min(cfg.risk.vol_target / vol, 2.0)
            else:
                s = 1.0

            # --- Layer 3: Drawdown multiplier ---
            dd = (equity - peak_equity) / peak_equity  # ≤ 0
            m = 1.0 + dd / abs(cfg.risk.max_drawdown_halt)
            m = max(m, 0.0)  # floor at 0 (beyond max_dd → circuit fires)

            # --- Layer 4: Regime multiplier ---
            is_bear = bool(regime_is_bear[i])
            conf = float(regime_conf[i])
            if is_bear and conf >= cfg.regime.regime_confidence_threshold:
                r = 0.0
            elif int(regime_id_arr[i]) == 1:  # neutral regime
                r = cfg.regime.regime_neutral_multiplier
            else:
                r = 1.0

            raw_target = float(np.clip(f * s * m * r, 0.0, 1.0))

            # --- Fix 4: per-bar diagnostic (debug level) ---
            if f == 0.0:
                stats.killed_by_kelly += 1

            # --- Layer 5: Circuit-Breaker Bypass (BEFORE hysteresis) ---
            if r == 0.0 or m <= 0.0:
                # Emergency exit — bypass hysteresis unconditionally
                if current_position != 0.0:
                    cost = abs(current_position) * flat_fee
                    equity *= 1.0 - cost
                current_position = 0.0
                cb_fired_out[i] = True
                if r == 0.0:
                    stats.killed_by_regime_cb += 1
                if m <= 0.0:
                    stats.killed_by_dd_cb += 1
                stats.circuit_breaker_activations += 1
                position_out[i] = 0.0
                continue

            # --- Layer 6: Hysteresis Gate (ASYMMETRIC for re-entry) ---
            # Fix 1: after CB flattens to 0%, use a much smaller threshold
            # to allow re-entry.  Standard threshold only applies to
            # position *changes* when already invested.
            if current_position == 0.0:
                # Re-entry from flat: any non-trivial target passes
                effective_threshold = cfg.risk.rebalance_threshold * 0.2
            else:
                effective_threshold = cfg.risk.rebalance_threshold

            delta = abs(raw_target - current_position)
            if delta < effective_threshold:
                hysteresis_out[i] = True
                stats.suppressed_by_hysteresis += 1
                position_out[i] = current_position
                continue

            # Rebalance
            if current_position == 0.0 and raw_target > 0.0:
                stats.re_entries_after_flatten += 1
            cost = delta * flat_fee
            equity *= 1.0 - cost
            current_position = raw_target
            stats.total_rebalance_events += 1
            position_out[i] = current_position

    except RiskError:
        raise
    except Exception as exc:
        raise RiskError(f"Position sizing failed at bar {i}: {exc}") from exc

    logger.info(
        "Phase 7 — Risk sizing complete | rebalances=%d | hysteresis_blocked=%d "
        "| circuit_breaker=%d | suppression_rate=%.1f%%",
        stats.total_rebalance_events,
        stats.suppressed_by_hysteresis,
        stats.circuit_breaker_activations,
        stats.suppression_rate * 100,
    )
    logger.info(
        "  Kill breakdown: kelly_no_edge=%d | regime_cb=%d | dd_cb=%d | re_entries=%d",
        stats.killed_by_kelly,
        stats.killed_by_regime_cb,
        stats.killed_by_dd_cb,
        stats.re_entries_after_flatten,
    )

    if stats.suppression_rate > 0.60:
        logger.warning(
            "Hysteresis suppression rate %.1f%% > 60%%. "
            "The rebalance_threshold may be too aggressive.",
            stats.suppression_rate * 100,
        )

    df_out = df.copy()
    df_out["Position_Size"] = position_out
    df_out["circuit_breaker_fired"] = cb_fired_out
    df_out["hysteresis_blocked"] = hysteresis_out

    return df_out, stats

"""
Phase 8 — Execution Cost Simulation.

Models realistic transaction friction for each rebalance event:

  Spread  = base_bps + vol_sensitivity × HL_ratio × vol_ratio
  Impact  = η × σ × √(order_fraction)        (Almgren-Chriss sqrt impact)
  Fill    = min(desired_delta, max_fill_fraction × bar_dollar_volume / portfolio)

All costs are expressed as a fraction of portfolio value (not BPS) and stored
in the 'execution_cost_pct' column.  The total cost for a bar is the spread +
impact cost applied to the filled portion of the desired order.

Notes:
  - Bars with no position change have zero execution cost.
  - Partial fills leave a residual that is filled in subsequent bars (clipped
    to the bar's fill capacity — the backtest sees the smoothed position).
  - If execution is disabled, a flat fee (trading_fee_bps + slippage_bps) is
    used instead.  backtest.py selects the appropriate cost series automatically.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import ExecutionError

logger = logging.getLogger(__name__)


def simulate_execution(df: pd.DataFrame, cfg: ArgusConfig) -> pd.DataFrame:
    """
    Compute per-bar execution costs and adjusted position sizes after partial fills.

    Requires columns: Position_Size, Close, High, Low, Volume, realised_vol_20d.

    Adds columns:
      execution_cost_pct   — total cost as fraction of portfolio value
      filled_position      — actual position after fill constraint

    Returns the augmented DataFrame.  Raises ExecutionError on failure.
    """
    if not cfg.execution.enabled:
        logger.info("Phase 8 — Execution simulation disabled; using flat fee.")
        flat = (cfg.backtest.trading_fee_bps + cfg.backtest.slippage_bps) / 10_000.0
        result = df.copy()
        pos = df["Position_Size"].values
        changes = np.abs(np.diff(pos, prepend=0.0))
        result["execution_cost_pct"] = changes * flat
        result["filled_position"] = pos
        return result

    required = ["Position_Size", "Close", "High", "Low", "Volume", "realised_vol_20d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ExecutionError(f"Missing columns for execution simulation: {missing}")

    logger.info("Phase 8 — Simulating execution costs (spread + Almgren-Chriss impact).")

    try:
        n = len(df)
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values
        vol_20d = df["realised_vol_20d"].values
        desired_pos = df["Position_Size"].values

        # Pre-compute vol_ratio (current vol relative to 60-bar average)
        vol_series = pd.Series(vol_20d)
        avg_vol_60 = vol_series.rolling(60, min_periods=1).mean().values

        cost_out = np.zeros(n)
        filled_pos = np.zeros(n)
        current_position = 0.0

        # Estimate portfolio value for fill capacity (use starting_balance × equity multiplier)
        # We approximate with a running equity estimate
        portfolio = float(cfg.backtest.starting_balance)
        flat_fee = (cfg.backtest.trading_fee_bps + cfg.backtest.slippage_bps) / 10_000.0

        for i in range(n):
            if i > 0 and close[i - 1] > 0:
                bar_ret = (close[i] - close[i - 1]) / close[i - 1]
                portfolio *= 1.0 + current_position * bar_ret

            desired_delta = desired_pos[i] - current_position

            if abs(desired_delta) < 1e-9:
                filled_pos[i] = current_position
                continue

            # Apply execution delay
            if cfg.execution.delay_bars > 0 and i < cfg.execution.delay_bars:
                filled_pos[i] = current_position
                continue

            # --- Fill capacity (Almgren-Chriss): max 2% of bar dollar-volume ---
            bar_dollar_vol = volume[i] * close[i]
            if bar_dollar_vol > 0 and portfolio > 0:
                max_delta = cfg.execution.max_fill_fraction * bar_dollar_vol / portfolio
            else:
                max_delta = abs(desired_delta)

            actual_delta = np.sign(desired_delta) * min(abs(desired_delta), max_delta)

            # --- Spread cost ---
            hl_ratio = (high[i] - low[i]) / close[i] if close[i] > 0 else 0.0
            v_ratio = vol_20d[i] / avg_vol_60[i] if avg_vol_60[i] > 0 else 1.0
            spread_bps = (
                cfg.execution.base_spread_bps
                + cfg.execution.vol_sensitivity * hl_ratio * v_ratio * 1_000.0  # HL_ratio ≈ 1-5%
            )
            spread_cost = (spread_bps / 10_000.0) * abs(actual_delta)

            # --- Market impact (Almgren-Chriss sqrt model) ---
            sigma = vol_20d[i] / np.sqrt(cfg.backtest.trading_days_per_year)  # daily vol
            order_frac = (
                abs(actual_delta) * portfolio / bar_dollar_vol
                if bar_dollar_vol > 0
                else 0.0
            )
            impact_cost = cfg.execution.impact_coefficient * sigma * np.sqrt(order_frac) * abs(actual_delta)

            total_cost = spread_cost + impact_cost + flat_fee * abs(actual_delta)

            current_position += actual_delta
            cost_out[i] = total_cost
            filled_pos[i] = current_position

        logger.debug(
            "Execution simulation: total cost = %.4f%% of portfolio, avg per rebalance = %.4f%%",
            cost_out.sum() * 100,
            cost_out[cost_out > 0].mean() * 100 if (cost_out > 0).any() else 0,
        )

        result = df.copy()
        result["execution_cost_pct"] = cost_out
        result["filled_position"] = filled_pos
        return result

    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError(f"Execution simulation failed: {exc}") from exc

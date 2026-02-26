"""
Phase 2 — Feature Engineering.

Implements 9 scale-invariant technical indicators as pure functions (no pandas-ta)
and a triple-barrier labeller that records PT/SL distances for downstream dynamic Kelly.

Output columns added to the DataFrame
--------------------------------------
Technical (9):
  RSI, EMA_cross, MACD_hist, ATR_ratio, BB_pctb,
  Vol_regime, OBV_slope, RVOL, Vol_price_div

Ancillary (used by risk.py):
  realised_vol_20d   — annualised 20-day realised volatility

Triple-barrier (per bar):
  Target             — 1 if PT hit first, 0 if SL/expiry
  hit_type           — "pt" | "sl" | "expiry"
  pnl_pct            — actual % return at exit
  pt_distance        — today's PT threshold (for dynamic Kelly)
  sl_distance        — today's SL threshold (for dynamic Kelly)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Feature column lists (used by model.py / ensemble.py)
# -------------------------------------------------------------------
TECHNICAL_FEATURE_COLS = [
    "RSI",
    "EMA_cross",
    "MACD_hist",
    "ATR_ratio",
    "BB_pctb",
    "Vol_regime",
    "OBV_slope",
    "RVOL",
    "Vol_price_div",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of model-input feature columns present in *df*.

    Includes technical features plus any on-chain and regime columns that
    are already attached (onchain.py and regime.py must run first if desired).
    """
    onchain_synthetic = [
        "MVRV_Z_synthetic",
        "Exchange_Flow_synthetic",
        "Funding_Rate_synthetic",
        "SOPR_synthetic",
    ]
    onchain_real = ["MVRV_Z", "Exchange_Flow", "Funding_Rate", "SOPR"]
    regime_cols = ["Regime", "Regime_Confidence", "Regime_Is_Bear"]

    candidates = (
        TECHNICAL_FEATURE_COLS
        + onchain_synthetic
        + onchain_real
        + regime_cols
    )
    return [c for c in candidates if c in df.columns]


# -------------------------------------------------------------------
# Pure-function indicator helpers
# -------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI via EMA (alpha = 1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Normalised linear-regression slope over a rolling window (raw=True, O(n·w))."""
    x = np.arange(window, dtype=float)
    x -= x.mean()
    x_sq = float((x**2).sum())

    def _slope(y: np.ndarray) -> float:
        if x_sq == 0:
            return 0.0
        return float(np.dot(x, y - y.mean()) / x_sq)

    return series.rolling(window).apply(_slope, raw=True)


def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    mu = series.rolling(lookback).mean()
    sigma = series.rolling(lookback).std()
    return (series - mu) / sigma.replace(0, np.nan)


# -------------------------------------------------------------------
# 9 scale-invariant features
# -------------------------------------------------------------------

def _compute_technical_features(df: pd.DataFrame, cfg: ArgusConfig) -> pd.DataFrame:
    f = cfg.features
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    returns = close.pct_change()

    out = pd.DataFrame(index=df.index)

    # 1. RSI — overbought / oversold (scale: 0–100, already relative)
    out["RSI"] = _rsi(close, f.rsi_period)

    # 2. EMA_cross — short vs long momentum, normalised by long EMA
    ema_s = _ema(close, f.ema_short)
    ema_l = _ema(close, f.ema_long)
    out["EMA_cross"] = (ema_s - ema_l) / ema_l.replace(0, np.nan)

    # 3. MACD_hist — momentum acceleration, normalised by close
    macd_line = ema_s - ema_l
    signal_line = _ema(macd_line, f.macd_signal)
    out["MACD_hist"] = (macd_line - signal_line) / close.replace(0, np.nan)

    # 4. ATR_ratio — normalised chaos: ATR / close
    atr = _atr(high, low, close, f.atr_period)
    out["ATR_ratio"] = atr / close.replace(0, np.nan)

    # 5. BB_pctb — position within Bollinger bands
    bb_mid = close.rolling(f.bb_period).mean()
    bb_std = close.rolling(f.bb_period).std()
    bb_upper = bb_mid + f.bb_std * bb_std
    bb_lower = bb_mid - f.bb_std * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    out["BB_pctb"] = (close - bb_lower) / band_width

    # 6. Vol_regime — 20d realised vol / 60d realised vol, z-scored over 120d
    vol_20 = returns.rolling(f.vol_regime_short).std() * np.sqrt(365)
    vol_60 = returns.rolling(f.vol_regime_long).std() * np.sqrt(365)
    vol_ratio = vol_20 / vol_60.replace(0, np.nan)
    out["Vol_regime"] = _zscore(vol_ratio, f.vol_regime_lookback)

    # 7. OBV_slope — accumulation / distribution trend, z-scored
    obv = (np.sign(close.diff()) * volume).cumsum()
    raw_slope = _rolling_slope(obv, f.obv_slope_period)
    out["OBV_slope"] = _zscore(raw_slope, f.obv_slope_lookback)

    # 8. RVOL — current vs average volume activity
    avg_vol = volume.rolling(f.rvol_period).mean()
    out["RVOL"] = volume / avg_vol.replace(0, np.nan)

    # 9. Vol_price_div — volume confirms or contradicts price
    #    = z-scored(vol momentum) − z-scored(price momentum)
    price_mom = close.pct_change(f.vol_price_div_period)
    vol_mom = volume.pct_change(f.vol_price_div_period)
    price_z = _zscore(price_mom, f.vol_price_div_lookback)
    vol_z = _zscore(vol_mom, f.vol_price_div_lookback)
    out["Vol_price_div"] = vol_z - price_z

    # Ancillary: annualised 20-day realised vol (used by risk.py VolTgt)
    out["realised_vol_20d"] = returns.rolling(f.barrier_vol_period).std() * np.sqrt(365)

    return out


# -------------------------------------------------------------------
# Triple-barrier labeller
# -------------------------------------------------------------------

def _triple_barrier_label(
    df: pd.DataFrame, cfg: ArgusConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each bar compute:
      target       — 1 if PT hit first, else 0 (SL or expiry)
      hit_type     — "pt" | "sl" | "expiry"
      pnl_pct      — % return at exit relative to entry
      pt_distance  — PT barrier threshold (% gain) at that bar's volatility
      sl_distance  — SL barrier threshold (% loss) at that bar's volatility
    """
    f = cfg.features
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    daily_returns = pd.Series(close).pct_change()
    daily_vol = daily_returns.rolling(f.barrier_vol_period).std().values

    target = np.zeros(n, dtype=int)
    hit_type = np.full(n, "expiry", dtype=object)
    pnl_pct = np.zeros(n, dtype=float)
    pt_dist_arr = np.zeros(n, dtype=float)
    sl_dist_arr = np.zeros(n, dtype=float)

    for i in range(n):
        vol = daily_vol[i]
        if np.isnan(vol) or vol == 0.0:
            continue

        pt_d = float(np.exp(f.barrier_pt_multiplier * vol) - 1.0)
        sl_d = float(1.0 - np.exp(-f.barrier_sl_multiplier * vol))
        pt_dist_arr[i] = pt_d
        sl_dist_arr[i] = sl_d

        entry = close[i]
        pt_level = entry * (1.0 + pt_d)
        sl_level = entry * (1.0 - sl_d)
        horizon = min(f.barrier_horizon, n - i - 1)

        for j in range(1, horizon + 1):
            if high[i + j] >= pt_level:
                target[i] = 1
                hit_type[i] = "pt"
                pnl_pct[i] = (pt_level - entry) / entry
                break
            if low[i + j] <= sl_level:
                target[i] = 0
                hit_type[i] = "sl"
                pnl_pct[i] = (sl_level - entry) / entry
                break
        else:
            # Expiry: actual close return over horizon
            if horizon > 0:
                pnl_pct[i] = (close[i + horizon] - entry) / entry
            target[i] = 0  # expiry = class 0 per blueprint

    return target, hit_type, pnl_pct, pt_dist_arr, sl_dist_arr


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def generate_features(df: pd.DataFrame, cfg: ArgusConfig) -> pd.DataFrame:
    """
    Compute all 9 technical features, realised_vol_20d, and triple-barrier labels.

    Raises FeatureEngineeringError on any failure.
    """
    logger.info("Phase 2 — Engineering features on %d bars.", len(df))

    try:
        tech = _compute_technical_features(df, cfg)
        result = df.copy()
        for col in tech.columns:
            result[col] = tech[col].values

        target, hit_type, pnl_pct, pt_dist, sl_dist = _triple_barrier_label(result, cfg)
        result["Target"] = target
        result["hit_type"] = hit_type
        result["pnl_pct"] = pnl_pct
        result["pt_distance"] = pt_dist
        result["sl_distance"] = sl_dist

        # Drop warm-up rows where indicators are NaN
        before = len(result)
        result.dropna(subset=TECHNICAL_FEATURE_COLS + ["realised_vol_20d"], inplace=True)
        dropped = before - len(result)
        logger.info(
            "Dropped %d warm-up rows. %d bars remain after feature engineering.",
            dropped,
            len(result),
        )

        result.to_csv(cfg.data.processed_data_path)
        logger.info("Processed data saved to %s", cfg.data.processed_data_path)
        return result

    except FeatureEngineeringError:
        raise
    except Exception as exc:
        raise FeatureEngineeringError(
            f"Feature engineering failed: {exc}"
        ) from exc

"""
Phase 3 — On-Chain Feature Proxies (Experimental).

When real API sources (Glassnode, CryptoQuant) are unavailable, four synthetic
proxies are derived from price and volume.  All synthetic columns carry a
'_synthetic' suffix and a runtime warning is logged on every run.

Stubs for real data sources are provided as _fetch_real_*() returning None;
connect them to a live API to disable the synthetic fallback.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import OnChainError

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Stubs — replace with real API calls to disable synthetic fallback
# -------------------------------------------------------------------

def _fetch_real_mvrv_z(cfg: ArgusConfig) -> pd.Series | None:  # noqa: ARG001
    return None


def _fetch_real_exchange_flow(cfg: ArgusConfig) -> pd.Series | None:  # noqa: ARG001
    return None


def _fetch_real_funding_rate(cfg: ArgusConfig) -> pd.Series | None:  # noqa: ARG001
    return None


def _fetch_real_sopr(cfg: ArgusConfig) -> pd.Series | None:  # noqa: ARG001
    return None


# -------------------------------------------------------------------
# Synthetic proxy implementations
# -------------------------------------------------------------------

def _synthetic_mvrv_z(df: pd.DataFrame) -> pd.Series:
    """
    MVRV-Z proxy: z-score of (price / 200-day SMA - 1).
    Captures how far price has deviated from long-run 'realised value'.
    """
    sma200 = df["Close"].rolling(200).mean()
    ratio = (df["Close"] / sma200.replace(0, np.nan)) - 1.0
    mu = ratio.rolling(365).mean()
    sigma = ratio.rolling(365).std()
    return (ratio - mu) / sigma.replace(0, np.nan)


def _synthetic_exchange_flow(df: pd.DataFrame) -> pd.Series:
    """
    Exchange flow proxy: z-scored ratio of current volume to 30-day average.
    Spikes in volume relative to average suggest exchange inflows/outflows.
    """
    avg_vol = df["Volume"].rolling(30).mean()
    ratio = df["Volume"] / avg_vol.replace(0, np.nan)
    mu = ratio.rolling(90).mean()
    sigma = ratio.rolling(90).std()
    return (ratio - mu) / sigma.replace(0, np.nan)


def _synthetic_funding_rate(df: pd.DataFrame) -> pd.Series:
    """
    Funding rate proxy: z-scored 7-day momentum scaled by realised volatility.
    High positive momentum + low vol → crowded long positioning.
    """
    mom_7 = df["Close"].pct_change(7)
    vol_20 = df["Close"].pct_change().rolling(20).std()
    # Scale momentum by inverse volatility: crowded when momentum/vol is high
    scaled = mom_7 / vol_20.replace(0, np.nan)
    return scaled.rolling(60).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() or np.nan), raw=True
    )


def _synthetic_sopr(df: pd.DataFrame) -> pd.Series:
    """
    SOPR proxy: ratio of today's close to close N days ago (realised price approx).
    Values > 1 → holders selling at a profit; < 1 → selling at a loss.
    """
    realised_price = df["Close"].shift(90)  # 90-day lag as rough cost-basis proxy
    sopr_raw = df["Close"] / realised_price.replace(0, np.nan)
    # Z-score so model sees a stationary, scale-invariant signal
    mu = sopr_raw.rolling(120).mean()
    sigma = sopr_raw.rolling(120).std()
    return (sopr_raw - mu) / sigma.replace(0, np.nan)


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------

def add_onchain_features(df: pd.DataFrame, cfg: ArgusConfig) -> pd.DataFrame:
    """
    Attach four on-chain proxy columns to *df*.

    When cfg.onchain.experimental is True (default), synthetic proxies are used
    and a warning is emitted.  Returns the augmented DataFrame.

    Raises OnChainError on any computation failure.
    """
    if not cfg.onchain.enabled:
        logger.info("Phase 3 — On-chain features disabled; skipping.")
        return df

    logger.info("Phase 3 — Computing on-chain feature proxies.")

    try:
        result = df.copy()

        features: dict[str, tuple[callable, callable, str, str]] = {
            "MVRV_Z": (_fetch_real_mvrv_z, _synthetic_mvrv_z, "MVRV_Z", "MVRV_Z_synthetic"),
            "Exchange_Flow": (
                _fetch_real_exchange_flow,
                _synthetic_exchange_flow,
                "Exchange_Flow",
                "Exchange_Flow_synthetic",
            ),
            "Funding_Rate": (
                _fetch_real_funding_rate,
                _synthetic_funding_rate,
                "Funding_Rate",
                "Funding_Rate_synthetic",
            ),
            "SOPR": (_fetch_real_sopr, _synthetic_sopr, "SOPR", "SOPR_synthetic"),
        }

        for name, (real_fn, synth_fn, real_col, synth_col) in features.items():
            real_data: pd.Series | None = None

            if not cfg.onchain.experimental:
                real_data = real_fn(cfg)

            if real_data is not None:
                result[real_col] = real_data.reindex(result.index)
                logger.info("  %s: using real data source.", name)
            else:
                if not cfg.onchain.use_synthetic_fallback:
                    raise OnChainError(
                        f"No real data for {name!r} and synthetic fallback is disabled."
                    )
                logger.warning(
                    "  %s: real source unavailable — using SYNTHETIC proxy (%s). "
                    "These are nonlinear price/volume transforms and may not add "
                    "independent information. Connect a real API to remove this warning.",
                    name,
                    synth_col,
                )
                result[synth_col] = synth_fn(df).values

        if cfg.onchain.experimental:
            logger.warning(
                "Phase 3 — All on-chain features are SYNTHETIC (experimental=True). "
                "Real API sources (Glassnode, CryptoQuant) have not been configured."
            )

        return result

    except OnChainError:
        raise
    except Exception as exc:
        raise OnChainError(f"On-chain feature computation failed: {exc}") from exc

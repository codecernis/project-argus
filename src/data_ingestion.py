"""
Phase 1 — Data Ingestion.

Downloads BTC-USD OHLCV data from Yahoo Finance, flattens MultiIndex columns,
validates required columns, and saves a raw CSV for auditability.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.config import ArgusConfig
from src.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


def fetch_market_data(cfg: ArgusConfig) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.

    Returns a DataFrame indexed by date with columns [Open, High, Low, Close, Volume].
    Raises DataIngestionError on any failure.
    """
    logger.info(
        "Phase 1 — Fetching %s | period=%s | interval=%s",
        cfg.data.symbol,
        cfg.data.period,
        cfg.data.interval,
    )

    try:
        df = yf.download(
            tickers=cfg.data.symbol,
            period=cfg.data.period,
            interval=cfg.data.interval,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        raise DataIngestionError(
            f"yfinance download failed for {cfg.data.symbol!r}: {exc}"
        ) from exc

    if df is None or df.empty:
        raise DataIngestionError(
            f"No data returned for {cfg.data.symbol!r}. "
            "Check ticker symbol and network connection."
        )

    # Flatten MultiIndex columns produced by yfinance when multiple tickers are requested
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise column names (yfinance may return 'Adj Close' etc.)
    df.columns = [str(c).strip() for c in df.columns]

    # Drop any all-NaN rows that yfinance sometimes appends
    df.dropna(how="all", inplace=True)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise DataIngestionError(
            f"Downloaded data is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    df = df[sorted(_REQUIRED_COLS)].copy()  # keep only OHLCV, consistent order
    df.index.name = "Date"

    raw_path = cfg.data.raw_data_path
    df.to_csv(raw_path)
    logger.info(
        "Ingested %d rows (%s → %s). Raw snapshot saved to %s.",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
        raw_path,
    )

    return df

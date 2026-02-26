"""
Phase 9 — Backtesting Engine.

Computes vectorised equity curves for Argus and Buy & Hold, then reports:
  - 7 institutional risk metrics (Sharpe, Sortino, Calmar, CAGR, MaxDD, WinRate, ProfitFactor)
  - 3-tier cost sensitivity sweep (optimistic / base / pessimistic)
  - Per-fold feature importance drift summary
  - Hysteresis & circuit-breaker report

All metrics are computed under three cost regimes to stress-test robustness.
If Sharpe < 0.5 or ProfitFactor < 1.0 in the pessimistic tier, a warning is emitted.

Position column used:
  'filled_position' if present (execution sim ran), else 'Position_Size'.

Cost column used:
  'execution_cost_pct' if present, else flat fee derived from config.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config import ArgusConfig
from src.exceptions import BacktestError
from src.model import FoldMetadata
from src.risk import RiskStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost sweep tiers
# ---------------------------------------------------------------------------
COST_TIERS = {
    "optimistic": dict(spread_bps=3.0, impact_coeff=0.05, fee_bps=5.0, slippage_bps=3.0),
    "base":       dict(spread_bps=5.0, impact_coeff=0.10, fee_bps=10.0, slippage_bps=5.0),
    "pessimistic":dict(spread_bps=10.0,impact_coeff=0.20, fee_bps=15.0, slippage_bps=10.0),
}


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

@dataclass
class MetricSet:
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Sharpe":        self.sharpe,
            "Sortino":       self.sortino,
            "Calmar":        self.calmar,
            "CAGR":          self.cagr,
            "Max_DD":        self.max_dd,
            "Win_Rate":      self.win_rate,
            "Profit_Factor": self.profit_factor,
        }


def _compute_metrics(
    returns: pd.Series,
    trading_days: int = 365,
) -> MetricSet:
    """Compute all 7 institutional metrics from a daily returns series."""
    r = returns.dropna()
    if len(r) < 2:
        return MetricSet()

    mu = r.mean()
    sigma = r.std()
    n_years = len(r) / trading_days

    # CAGR
    terminal = (1.0 + r).prod()
    cagr = float(terminal ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0

    # Sharpe
    sharpe = float((mu / sigma) * np.sqrt(trading_days)) if sigma > 0 else 0.0

    # Sortino (downside deviation)
    downside = r[r < 0]
    sigma_down = downside.std()
    sortino = float((mu / sigma_down) * np.sqrt(trading_days)) if sigma_down > 0 else 0.0

    # Max drawdown
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd_series = (equity - peak) / peak
    max_dd = float(dd_series.min())

    # Calmar
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Win rate (profitable days / active days)
    active = r[r != 0.0]
    win_rate = float((active > 0).sum() / len(active)) if len(active) > 0 else 0.0

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return MetricSet(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        cagr=cagr,
        max_dd=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


# ---------------------------------------------------------------------------
# Equity curve simulation for the cost sweep
# ---------------------------------------------------------------------------

def _simulate_equity_curve(
    position: np.ndarray,
    close: np.ndarray,
    fee_pct: float,
) -> np.ndarray:
    """
    Vectorised equity curve using a flat fee model.

    position[i] = fraction held from close of bar i onward.
    Bar-i return = position[i-1] × (close[i]/close[i-1] − 1) − rebalance_cost.
    """
    n = len(close)
    equity = np.ones(n)

    for i in range(1, n):
        if close[i - 1] <= 0:
            equity[i] = equity[i - 1]
            continue
        bar_ret = (close[i] - close[i - 1]) / close[i - 1]
        pos_return = position[i - 1] * bar_ret
        trade_cost = abs(position[i] - position[i - 1]) * fee_pct
        equity[i] = equity[i - 1] * (1.0 + pos_return - trade_cost)

    return equity


# ---------------------------------------------------------------------------
# Feature importance drift analysis
# ---------------------------------------------------------------------------

def _compute_importance_drift(
    fold_metadata: list[FoldMetadata],
) -> dict[str, Any]:
    """
    Compute importance drift statistics across all folds.

    Returns a dict with:
      mean_importance   : {feature: float}
      std_importance    : {feature: float}
      rank_first_fold   : {feature: int}
      rank_last_fold    : {feature: int}
      rank_drift        : {feature: int}   (|rank_last - rank_first|)
      unstable_features : list[str]        (rank drift > 5)
      dominant_features : list[str]        (mean importance > 30%)
    """
    if not fold_metadata:
        return {}

    all_features: set[str] = set()
    for fm in fold_metadata:
        all_features.update(fm.feature_importances.keys())

    if not all_features:
        return {}

    # Build (folds × features) matrix
    features = sorted(all_features)
    matrix = np.zeros((len(fold_metadata), len(features)))
    for fi, fm in enumerate(fold_metadata):
        for fj, feat in enumerate(features):
            matrix[fi, fj] = fm.feature_importances.get(feat, 0.0)

    mean_imp = matrix.mean(axis=0)
    std_imp = matrix.std(axis=0)

    def _ranks(row: np.ndarray) -> dict[str, int]:
        order = np.argsort(-row)  # descending
        return {features[idx]: int(rank + 1) for rank, idx in enumerate(order)}

    first_ranks = _ranks(matrix[0])
    last_ranks = _ranks(matrix[-1])

    drift = {f: abs(last_ranks.get(f, 0) - first_ranks.get(f, 0)) for f in features}
    unstable = [f for f in features if drift[f] > 5]
    dominant = [features[i] for i, m in enumerate(mean_imp) if m > 0.30]

    return {
        "mean_importance": dict(zip(features, mean_imp.tolist())),
        "std_importance": dict(zip(features, std_imp.tolist())),
        "rank_first_fold": first_ranks,
        "rank_last_fold": last_ranks,
        "rank_drift": drift,
        "unstable_features": unstable,
        "dominant_features": dominant,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    cfg: ArgusConfig,
    fold_metadata: list[FoldMetadata],
    risk_stats: RiskStats,
) -> dict[str, Any]:
    """
    Run the full backtest pipeline including cost sweep and reporting.

    Returns a results dict with equity curves, metrics, and diagnostics.
    Raises BacktestError on failure.
    """
    # Select position and cost columns
    pos_col = "filled_position" if "filled_position" in df.columns else "Position_Size"
    cost_col = "execution_cost_pct" if "execution_cost_pct" in df.columns else None

    required = [pos_col, "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BacktestError(f"Missing required columns for backtest: {missing}")

    # Only backtest bars with valid position data
    mask = df[pos_col].notna()
    if not mask.any():
        raise BacktestError("No bars with valid Position_Size — ensure WFO ran successfully.")

    bt_df = df[mask].copy()
    close = bt_df["Close"].values
    position = bt_df[pos_col].values
    n = len(bt_df)
    trading_days = cfg.backtest.trading_days_per_year

    logger.info(
        "Phase 9 — Backtesting %d bars | starting_balance=$%.0f",
        n,
        cfg.backtest.starting_balance,
    )

    try:
        # ----------------------------------------------------------------
        # Base backtest with execution-sim costs (or flat fee)
        # ----------------------------------------------------------------
        if cost_col:
            exec_cost = bt_df[cost_col].values
        else:
            flat_fee = (cfg.backtest.trading_fee_bps + cfg.backtest.slippage_bps) / 10_000.0
            delta_pos = np.abs(np.diff(position, prepend=0.0))
            exec_cost = delta_pos * flat_fee

        # Argus equity curve with execution costs
        argus_equity = np.ones(n) * cfg.backtest.starting_balance
        for i in range(1, n):
            if close[i - 1] <= 0:
                argus_equity[i] = argus_equity[i - 1]
                continue
            bar_ret = (close[i] - close[i - 1]) / close[i - 1]
            argus_equity[i] = argus_equity[i - 1] * (
                1.0 + position[i - 1] * bar_ret - exec_cost[i]
            )

        argus_returns = pd.Series(
            np.diff(argus_equity, prepend=argus_equity[0]) / argus_equity,
            index=bt_df.index,
        )

        # Buy & Hold
        bh_returns = pd.Series(
            (close[1:] - close[:-1]) / close[:-1],
            index=bt_df.index[1:],
        )

        argus_metrics = _compute_metrics(argus_returns, trading_days)
        bh_metrics = _compute_metrics(bh_returns, trading_days)

        # ----------------------------------------------------------------
        # Cost sensitivity sweep
        # ----------------------------------------------------------------
        sweep_results: dict[str, MetricSet] = {}
        if cfg.backtest.cost_sweep_enabled:
            for tier_name, tier_params in COST_TIERS.items():
                tier_fee = (tier_params["fee_bps"] + tier_params["slippage_bps"]) / 10_000.0
                tier_equity = _simulate_equity_curve(position, close, tier_fee)
                tier_equity *= cfg.backtest.starting_balance
                tier_ret = pd.Series(
                    np.diff(tier_equity, prepend=tier_equity[0]) / tier_equity,
                    index=bt_df.index,
                )
                sweep_results[tier_name] = _compute_metrics(tier_ret, trading_days)

        # ----------------------------------------------------------------
        # Feature importance drift
        # ----------------------------------------------------------------
        drift_report = _compute_importance_drift(fold_metadata)

        # ----------------------------------------------------------------
        # Save results CSV
        # ----------------------------------------------------------------
        bt_df["Argus_Equity"] = argus_equity
        bh_equity_curve = cfg.backtest.starting_balance * (1.0 + bh_returns).cumprod()
        bt_df.loc[bh_returns.index, "BH_Equity"] = bh_equity_curve.values
        bt_df.to_csv(cfg.data.backtest_results_path)
        logger.info("Backtest results saved to %s", cfg.data.backtest_results_path)

        return {
            "argus_metrics": argus_metrics,
            "bh_metrics": bh_metrics,
            "sweep_results": sweep_results,
            "drift_report": drift_report,
            "risk_stats": risk_stats,
            "argus_equity": argus_equity,
            "bh_equity": bh_equity_curve if cfg.backtest.cost_sweep_enabled else None,
            "n_bars": n,
        }

    except BacktestError:
        raise
    except Exception as exc:
        raise BacktestError(f"Backtest computation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Formatted report printers (called by main.py)
# ---------------------------------------------------------------------------

def print_metrics_table(
    argus: MetricSet,
    bh: MetricSet,
    label: str = "BASE RESULTS",
) -> None:
    """Print a formatted side-by-side metrics comparison table."""
    w = 65
    sep = "─" * w
    print(f"\n{'═' * w}")
    print(f"  {label}")
    print(f"{'═' * w}")
    print(f"  {'Metric':<20} {'Argus':>15} {'Buy & Hold':>15}")
    print(sep)
    rows = [
        ("Sharpe",       f"{argus.sharpe:>+.3f}",       f"{bh.sharpe:>+.3f}"),
        ("Sortino",      f"{argus.sortino:>+.3f}",      f"{bh.sortino:>+.3f}"),
        ("Calmar",       f"{argus.calmar:>+.3f}",       f"{bh.calmar:>+.3f}"),
        ("CAGR",         f"{argus.cagr:>+.1%}",         f"{bh.cagr:>+.1%}"),
        ("Max DD",       f"{argus.max_dd:>+.1%}",       f"{bh.max_dd:>+.1%}"),
        ("Win Rate",     f"{argus.win_rate:>.1%}",      f"{bh.win_rate:>.1%}"),
        ("Profit Factor",f"{argus.profit_factor:>.3f}", f"{bh.profit_factor:>.3f}"),
    ]
    for metric, av, bv in rows:
        print(f"  {metric:<20} {av:>15} {bv:>15}")
    print(sep)


def print_cost_sweep_table(sweep: dict[str, MetricSet]) -> None:
    """Print the 3-tier cost sensitivity sweep table."""
    if not sweep:
        return
    w = 65
    print(f"\n{'═' * w}")
    print("  COST SENSITIVITY SWEEP")
    print(f"{'═' * w}")
    print(f"  {'Tier':<14} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'CAGR':>8} {'MaxDD':>8} {'PF':>7}")
    print("─" * w)
    for tier, m in sweep.items():
        flag = ""
        if m.sharpe < 0.5 or m.profit_factor < 1.0:
            flag = "  ⚠ WARNING"
        print(
            f"  {tier.capitalize():<14} {m.sharpe:>+8.3f} {m.sortino:>+8.3f} "
            f"{m.calmar:>+8.3f} {m.cagr:>+7.1%} {m.max_dd:>+7.1%} "
            f"{m.profit_factor:>7.3f}{flag}"
        )
        if flag:
            logger.warning(
                "Cost sweep '%s': Sharpe=%.3f < 0.5 or ProfitFactor=%.3f < 1.0. "
                "Strategy may not survive real-world friction.",
                tier,
                m.sharpe,
                m.profit_factor,
            )
    print("─" * w)


def print_importance_drift(drift: dict) -> None:
    """Print feature importance drift summary."""
    if not drift or "mean_importance" not in drift:
        return
    w = 65
    print(f"\n{'═' * w}")
    print("  FEATURE IMPORTANCE DRIFT SUMMARY")
    print(f"{'═' * w}")
    means = drift["mean_importance"]
    stds = drift["std_importance"]
    drifts = drift.get("rank_drift", {})
    sorted_feats = sorted(means, key=means.get, reverse=True)
    print(f"  {'Feature':<28} {'Mean':>8} {'Std':>8} {'RankDrift':>10}")
    print("─" * w)
    for feat in sorted_feats:
        d_flag = " [UNSTABLE]" if feat in drift.get("unstable_features", []) else ""
        print(
            f"  {feat:<28} {means[feat]:>8.4f} {stds.get(feat, 0):>8.4f} "
            f"{drifts.get(feat, 0):>10}{d_flag}"
        )
    if drift.get("dominant_features"):
        logger.warning(
            "Dominant features (>30%% avg importance): %s. Model may be over-reliant.",
            drift["dominant_features"],
        )
    print("─" * w)


def print_risk_stats(stats: RiskStats) -> None:
    """Print the hysteresis and circuit-breaker report."""
    w = 65
    print(f"\n{'═' * w}")
    print("  HYSTERESIS & CIRCUIT-BREAKER REPORT")
    print(f"{'═' * w}")
    print(f"  Total rebalance events  : {stats.total_rebalance_events:>10,}")
    print(f"  Suppressed by hysteresis: {stats.suppressed_by_hysteresis:>10,}")
    print(f"  Circuit-breaker fires   : {stats.circuit_breaker_activations:>10,}")
    print(f"  Suppression rate        : {stats.suppression_rate:>10.1%}")
    print(f"{'─' * w}")
    print("  KILL BREAKDOWN (why bars weren't traded)")
    print(f"{'─' * w}")
    print(f"  Kelly no-edge (f≤0)     : {stats.killed_by_kelly:>10,}")
    print(f"  Regime circuit-breaker   : {stats.killed_by_regime_cb:>10,}")
    print(f"  Drawdown circuit-breaker : {stats.killed_by_dd_cb:>10,}")
    print(f"  Re-entries after flatten : {stats.re_entries_after_flatten:>10,}")
    if stats.suppression_rate > 0.60:
        print("  ⚠  Suppression rate > 60% — rebalance_threshold may be too aggressive.")
    print("─" * w)

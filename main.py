"""
Project Argus v3.3 — Pipeline Orchestrator.

9 phases executed in sequence:
  1. Data Ingestion          (data_ingestion.py)
  2. Feature Engineering     (features.py)
  3. On-Chain Proxies        (onchain.py)
  4. Regime Detection        (regime.py)
  5. Hyperparameter Search   (optimizer.py)       [optional]
  6. WFO Model Training      (ensemble.py → model.py fallback)
  7. Position Sizing         (risk.py)
  8. Execution Simulation    (execution.py)
  9. Backtest & Reporting    (backtest.py)

A single try/except ArgusError wraps the entire pipeline.
At the end, four summary reports are printed:
  - Cost sensitivity sweep table
  - Feature importance drift table
  - Hysteresis & circuit-breaker report
  - Final metrics vs Buy & Hold
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import ArgusConfig
from src.exceptions import ArgusError
from src.data_ingestion import fetch_market_data
from src.features import generate_features
from src.onchain import add_onchain_features
from src.regime import add_regime_features
from src.optimizer import run_optimizer
from src.ensemble import train_and_predict_ensemble
from src.risk import compute_positions
from src.execution import simulate_execution
from src.backtest import (
    run_backtest,
    print_metrics_table,
    print_cost_sweep_table,
    print_importance_drift,
    print_risk_stats,
)

# ---------------------------------------------------------------------------
# Logging + simulation log file
# ---------------------------------------------------------------------------

LOGS_DIR = Path("logs")


class _TeeStream:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()


def _next_simulation_id() -> int:
    """Scan logs/ for argus_simulation_*.txt and return the next integer."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(LOGS_DIR.glob("argus_simulation_*.txt"))
    if not existing:
        return 1
    # Extract the highest N from argus_simulation_N.txt
    max_id = 0
    for p in existing:
        stem = p.stem  # e.g. "argus_simulation_12"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            max_id = max(max_id, int(parts[1]))
    return max_id + 1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("argus.main")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Open simulation log file and tee stdout ---
    sim_id = _next_simulation_id()
    log_path = LOGS_DIR / f"argus_simulation_{sim_id}.txt"
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
    sys.stdout = _TeeStream(sys.__stdout__, log_file)

    # Re-attach the root logging handler to the new tee'd stdout
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout

    banner = "=" * 65
    print(f"\n{banner}")
    print("   PROJECT ARGUS v3.3  —  Quantitative BTC-USD Backtesting")
    print(f"   Simulation #{sim_id}  —  Log: {log_path}")
    print(f"{banner}\n")

    cfg = ArgusConfig()

    try:
        # ----------------------------------------------------------------
        # Phase 1: Data Ingestion
        # ----------------------------------------------------------------
        logger.info("▶ Phase 1/9 — Data Ingestion")
        raw_df = fetch_market_data(cfg)

        # ----------------------------------------------------------------
        # Phase 2: Feature Engineering
        # ----------------------------------------------------------------
        logger.info("▶ Phase 2/9 — Feature Engineering")
        feat_df = generate_features(raw_df, cfg)

        # ----------------------------------------------------------------
        # Phase 3: On-Chain Proxies
        # ----------------------------------------------------------------
        logger.info("▶ Phase 3/9 — On-Chain Proxies")
        onchain_df = add_onchain_features(feat_df, cfg)

        # ----------------------------------------------------------------
        # Phase 4: Regime Detection
        # ----------------------------------------------------------------
        logger.info("▶ Phase 4/9 — Regime Detection")
        regime_df = add_regime_features(onchain_df, cfg)

        # ----------------------------------------------------------------
        # Phase 5: Hyperparameter Optimisation (optional)
        # ----------------------------------------------------------------
        logger.info("▶ Phase 5/9 — Hyperparameter Optimisation")
        cfg = run_optimizer(raw_df, cfg)

        # ----------------------------------------------------------------
        # Phase 6: Walk-Forward Model Training (ensemble or RF fallback)
        # ----------------------------------------------------------------
        logger.info("▶ Phase 6/9 — Walk-Forward Model Training")
        model_df, fold_metadata = train_and_predict_ensemble(regime_df, cfg)

        # Drop bars with no predictions (before the first test window)
        model_df = model_df[model_df["Probability"].notna()].copy()
        logger.info("%d bars with model predictions.", len(model_df))

        # ----------------------------------------------------------------
        # Phase 7: Position Sizing (Kelly + layers + circuit breaker)
        # ----------------------------------------------------------------
        logger.info("▶ Phase 7/9 — Dynamic Kelly Position Sizing")
        sized_df, risk_stats = compute_positions(model_df, cfg)

        # ----------------------------------------------------------------
        # Phase 8: Execution Cost Simulation
        # ----------------------------------------------------------------
        logger.info("▶ Phase 8/9 — Execution Cost Simulation")
        exec_df = simulate_execution(sized_df, cfg)

        # ----------------------------------------------------------------
        # Phase 9: Backtest & Reporting
        # ----------------------------------------------------------------
        logger.info("▶ Phase 9/9 — Backtesting & Reporting")
        results = run_backtest(exec_df, cfg, fold_metadata, risk_stats)

    except ArgusError as exc:
        logger.critical("Pipeline halted: %s", exc)
        sys.stdout = sys.__stdout__
        log_file.close()
        logger.info("Simulation log saved to %s", log_path)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Summary Reports
    # ----------------------------------------------------------------
    argus_m = results["argus_metrics"]
    bh_m = results["bh_metrics"]
    sweep = results["sweep_results"]
    drift = results["drift_report"]
    stats = results["risk_stats"]

    print_metrics_table(argus_m, bh_m, label="ARGUS vs BUY & HOLD — BASE RESULTS")
    print_cost_sweep_table(sweep)
    print_importance_drift(drift)
    print_risk_stats(stats)

    print(f"\n{'=' * 65}")
    print("   PROJECT ARGUS v3.3  —  PIPELINE COMPLETE")
    print(f"{'=' * 65}\n")

    # --- Close simulation log ---
    sys.stdout = sys.__stdout__
    log_file.close()
    logger.info("Simulation #%d log saved to %s", sim_id, log_path)


if __name__ == "__main__":
    main()

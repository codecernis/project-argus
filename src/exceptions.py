"""
Typed exception hierarchy for Project Argus v3.3.

Every module raises a specific subclass of ArgusError.
The orchestrator catches ArgusError at the top level.
"""


class ArgusError(Exception):
    """Base exception for all Project Argus errors."""


class DataIngestionError(ArgusError):
    """Raised by data_ingestion.py when fetch or validation fails."""


class FeatureEngineeringError(ArgusError):
    """Raised by features.py when indicator computation or labelling fails."""


class OnChainError(ArgusError):
    """Raised by onchain.py when proxy computation fails."""


class RegimeError(ArgusError):
    """Raised by regime.py when HMM training or prediction fails."""


class ModelError(ArgusError):
    """Raised by model.py when WFO training or prediction fails."""


class EnsembleError(ArgusError):
    """Raised by ensemble.py when blender training or prediction fails."""


class RiskError(ArgusError):
    """Raised by risk.py when position sizing computation fails."""


class ExecutionError(ArgusError):
    """Raised by execution.py when execution cost simulation fails."""


class BacktestError(ArgusError):
    """Raised by backtest.py when equity curve or metric computation fails."""


class OptimizerError(ArgusError):
    """Raised by optimizer.py when hyperparameter search fails."""

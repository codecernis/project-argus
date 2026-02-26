"""
Configuration module for Project Argus v3.3.

All settings live in frozen dataclasses validated at construction time.
Pass an ArgusConfig instance to every module function — no global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataSettings:
    symbol: str = "BTC-USD"
    period: str = "max"
    interval: str = "1d"
    data_dir: str = "data"

    def __post_init__(self) -> None:
        if self.interval not in ("1d", "1h", "1wk"):
            raise ValueError(f"Unsupported interval: {self.interval!r}")

    @property
    def raw_data_path(self) -> Path:
        return Path(self.data_dir) / "raw_data.csv"

    @property
    def processed_data_path(self) -> Path:
        return Path(self.data_dir) / "argus_market_data.csv"

    @property
    def backtest_results_path(self) -> Path:
        return Path(self.data_dir) / "backtest_results.csv"


@dataclass(frozen=True)
class FeatureSettings:
    # RSI
    rsi_period: int = 14
    # EMA cross
    ema_short: int = 12
    ema_long: int = 26
    # MACD
    macd_signal: int = 9
    # ATR
    atr_period: int = 14
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # Vol regime
    vol_regime_short: int = 20
    vol_regime_long: int = 60
    vol_regime_lookback: int = 120
    # OBV slope
    obv_slope_period: int = 10
    obv_slope_lookback: int = 60
    # RVOL
    rvol_period: int = 20
    # Vol-price divergence
    vol_price_div_period: int = 5
    vol_price_div_lookback: int = 60
    # Triple-barrier target
    barrier_horizon: int = 10
    barrier_pt_multiplier: float = 2.0
    barrier_sl_multiplier: float = 1.0
    barrier_vol_period: int = 20

    def __post_init__(self) -> None:
        if self.barrier_pt_multiplier <= 0:
            raise ValueError("barrier_pt_multiplier must be positive")
        if self.barrier_sl_multiplier <= 0:
            raise ValueError("barrier_sl_multiplier must be positive")
        if self.barrier_horizon < 1:
            raise ValueError("barrier_horizon must be >= 1")


@dataclass(frozen=True)
class OnChainSettings:
    enabled: bool = True
    experimental: bool = True
    use_synthetic_fallback: bool = True

    def __post_init__(self) -> None:
        if self.enabled and not self.experimental and not self.use_synthetic_fallback:
            raise ValueError(
                "OnChain enabled but experimental=False and use_synthetic_fallback=False. "
                "Configure a real data source or enable synthetic fallback."
            )


@dataclass(frozen=True)
class ModelSettings:
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 50
    random_state: int = 42
    train_window_days: int = 365
    step_size_days: int = 30
    purge_days: int = 10
    embargo_days: int = 5
    min_samples_to_features_ratio: float = 15.0
    time_decay_lambda: float = 0.0  # 0.0 = uniform; higher = more recent weight

    def __post_init__(self) -> None:
        gap = self.purge_days + self.embargo_days
        if self.train_window_days <= gap:
            raise ValueError(
                f"train_window_days ({self.train_window_days}) must be > "
                f"purge_days + embargo_days ({gap})"
            )
        if self.step_size_days < 1:
            raise ValueError("step_size_days must be >= 1")
        if self.min_samples_to_features_ratio <= 0:
            raise ValueError("min_samples_to_features_ratio must be positive")


@dataclass(frozen=True)
class EnsembleSettings:
    enabled: bool = True
    lgbm_n_estimators: int = 300
    lgbm_learning_rate: float = 0.05
    lgbm_max_depth: int = 6
    lgbm_num_leaves: int = 31
    meta_learner_burn_in: int = 1
    meta_learner_type: str = "scalar"  # only "scalar" supported in v3.3

    def __post_init__(self) -> None:
        if self.meta_learner_burn_in < 1:
            raise ValueError("meta_learner_burn_in must be >= 1")
        if self.meta_learner_type != "scalar":
            raise ValueError("Only 'scalar' meta_learner_type is supported in v3.3")
        if self.lgbm_learning_rate <= 0:
            raise ValueError("lgbm_learning_rate must be positive")


@dataclass(frozen=True)
class RegimeSettings:
    enabled: bool = True
    n_regimes: int = 3
    regime_confidence_threshold: float = 0.85
    regime_neutral_multiplier: float = 0.75
    window_type: str = "rolling"
    covariance_type: str = "full"
    n_iter: int = 100

    def __post_init__(self) -> None:
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        if not 0.0 < self.regime_confidence_threshold <= 1.0:
            raise ValueError("regime_confidence_threshold must be in (0, 1]")
        if not 0.0 < self.regime_neutral_multiplier <= 1.0:
            raise ValueError("regime_neutral_multiplier must be in (0, 1]")
        if self.window_type not in ("rolling", "expanding"):
            raise ValueError("window_type must be 'rolling' or 'expanding'")
        if self.covariance_type not in ("full", "diag", "tied", "spherical"):
            raise ValueError(f"Invalid covariance_type: {self.covariance_type!r}")


@dataclass(frozen=True)
class OptimizerSettings:
    enabled: bool = True
    n_trials: int = 50
    objective_metric: str = "sharpe"
    n_jobs: int = 1
    timeout: int | None = None
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.objective_metric not in ("sharpe", "sortino", "calmar"):
            raise ValueError(
                f"objective_metric must be 'sharpe', 'sortino', or 'calmar', "
                f"got {self.objective_metric!r}"
            )


@dataclass(frozen=True)
class RiskSettings:
    kelly_fraction: float = 0.40
    kelly_use_dynamic_odds: bool = True
    vol_target: float = 0.15
    max_drawdown_halt: float = -0.20
    rebalance_threshold: float = 0.05
    min_sl_distance: float = 1e-6  # degeneracy guard: b = pt/sl, avoid div-by-zero

    def __post_init__(self) -> None:
        if not 0.0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.vol_target <= 0:
            raise ValueError("vol_target must be positive")
        if self.max_drawdown_halt >= 0:
            raise ValueError("max_drawdown_halt must be negative")
        if self.rebalance_threshold < 0:
            raise ValueError("rebalance_threshold must be non-negative")


@dataclass(frozen=True)
class ExecutionSettings:
    enabled: bool = True
    base_spread_bps: float = 5.0
    vol_sensitivity: float = 1.0
    impact_coefficient: float = 0.1
    max_fill_fraction: float = 0.02  # max 2% of bar dollar-volume per order
    delay_bars: int = 0

    def __post_init__(self) -> None:
        if self.base_spread_bps < 0:
            raise ValueError("base_spread_bps must be non-negative")
        if self.impact_coefficient < 0:
            raise ValueError("impact_coefficient must be non-negative")
        if not 0.0 < self.max_fill_fraction <= 1.0:
            raise ValueError("max_fill_fraction must be in (0, 1]")
        if self.delay_bars < 0:
            raise ValueError("delay_bars must be non-negative")


@dataclass(frozen=True)
class BacktestSettings:
    starting_balance: float = 10000.0
    trading_fee_bps: float = 10.0
    slippage_bps: float = 5.0
    cost_sweep_enabled: bool = True
    trading_days_per_year: int = 365

    def __post_init__(self) -> None:
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        if self.trading_fee_bps < 0:
            raise ValueError("trading_fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")


@dataclass(frozen=True)
class ArgusConfig:
    data: DataSettings = field(default_factory=DataSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    onchain: OnChainSettings = field(default_factory=OnChainSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    ensemble: EnsembleSettings = field(default_factory=EnsembleSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)

    def __post_init__(self) -> None:
        # Ensure data directory exists at config construction time
        Path(self.data.data_dir).mkdir(parents=True, exist_ok=True)

"""Reproducible building blocks for adaptive signal-acquisition experiments."""

from .experiments import (
    AdaptiveStudyConfig,
    BaselineStudyConfig,
    run_adaptive_study,
    run_baseline_study,
)
from .reconstruction import LassoConfig, LassoResult, lasso_ista
from .signals import SignalConfig, generate_sparse_signal

__all__ = [
    "BaselineStudyConfig",
    "AdaptiveStudyConfig",
    "LassoConfig",
    "LassoResult",
    "SignalConfig",
    "generate_sparse_signal",
    "lasso_ista",
    "run_baseline_study",
    "run_adaptive_study",
]

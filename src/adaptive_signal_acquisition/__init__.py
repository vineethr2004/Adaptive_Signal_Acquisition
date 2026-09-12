"""Reproducible building blocks for adaptive signal-acquisition experiments."""

from .experiments import BaselineStudyConfig, run_baseline_study
from .reconstruction import LassoConfig, LassoResult, lasso_ista
from .signals import SignalConfig, generate_sparse_signal

__all__ = [
    "BaselineStudyConfig",
    "LassoConfig",
    "LassoResult",
    "SignalConfig",
    "generate_sparse_signal",
    "lasso_ista",
    "run_baseline_study",
]

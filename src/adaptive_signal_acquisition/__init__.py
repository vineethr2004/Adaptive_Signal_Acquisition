"""Reproducible building blocks for adaptive signal-acquisition experiments."""

from .experiments import (
    AdaptiveStudyConfig,
    BaselineStudyConfig,
    run_adaptive_study,
    run_baseline_study,
)
from .reconstruction import LassoConfig, LassoResult, lasso_ista
from .signals import SignalConfig, generate_sparse_signal
from .stage4 import Stage4AConfig, run_stage4a_diagnostics
from .stage4b import Stage4BConfig, run_stage4b_study

__all__ = [
    "BaselineStudyConfig",
    "AdaptiveStudyConfig",
    "LassoConfig",
    "LassoResult",
    "SignalConfig",
    "Stage4AConfig",
    "Stage4BConfig",
    "generate_sparse_signal",
    "lasso_ista",
    "run_baseline_study",
    "run_adaptive_study",
    "run_stage4a_diagnostics",
    "run_stage4b_study",
]

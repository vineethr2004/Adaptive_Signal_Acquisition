import numpy as np

from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.signals import SignalConfig
from adaptive_signal_acquisition.stage4 import (
    Stage4AConfig,
    gamma_method_name,
    run_stage4a_diagnostics,
)


def test_stage4a_ablation_is_paired_reproducible_and_diagnostic() -> None:
    lasso = LassoConfig(lambda_value=0.03, max_iterations=500)
    config = Stage4AConfig(
        signal=SignalConfig(dimension=16, sparsity=2),
        candidate_count=32,
        budgets=(4, 7),
        trials_per_budget=3,
        noise_std=0.05,
        lasso=lasso,
        provisional_lasso=lasso,
        prior_variance=0.125,
        gamma_values=(0.0, 0.5),
        bootstrap_resamples=100,
        seed=404,
    )
    first = run_stage4a_diagnostics(config)
    second = run_stage4a_diagnostics(config)

    method_count = 2 + len(config.gamma_values)
    assert len(first.records) == method_count * len(config.budgets) * config.trials_per_budget
    assert len(first.summaries) == method_count * len(config.budgets)
    assert len(first.diagnostic_traces) == (
        len(config.gamma_values) * config.trials_per_budget * max(config.budgets)
    )
    assert np.allclose(
        [record.nmse for record in first.records],
        [record.nmse for record in second.records],
    )

    gamma_zero = [
        trace
        for trace in first.diagnostic_traces
        if trace.method == gamma_method_name(0.0)
    ]
    gamma_half = [
        trace
        for trace in first.diagnostic_traces
        if trace.method == gamma_method_name(0.5)
    ]
    assert np.allclose([trace.lasso_proxy_component for trace in gamma_zero], 0.0)
    assert np.any(np.asarray([trace.lasso_proxy_component for trace in gamma_half]) > 0.0)
    assert np.all(np.isfinite([trace.provisional_nmse for trace in first.diagnostic_traces]))

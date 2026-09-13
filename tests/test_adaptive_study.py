import numpy as np

from adaptive_signal_acquisition.experiments import AdaptiveStudyConfig, run_adaptive_study
from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.sequential import InformationGuidedPolicyConfig
from adaptive_signal_acquisition.signals import SignalConfig


def test_adaptive_study_is_reproducible_and_records_audit_trace() -> None:
    lasso = LassoConfig(lambda_value=0.03, max_iterations=1_000)
    config = AdaptiveStudyConfig(
        signal=SignalConfig(dimension=20, sparsity=2),
        candidate_count=40,
        budgets=(5, 8),
        trials_per_budget=3,
        noise_std=0.05,
        lasso=lasso,
        policy=InformationGuidedPolicyConfig(
            noise_std=0.05, prior_variance=0.1, provisional_lasso=lasso
        ),
        seed=2027,
    )
    first = run_adaptive_study(config)
    second = run_adaptive_study(config)

    assert len(first.records) == 3 * 2 * 3
    assert len(first.summaries) == 3 * 2
    assert len(first.action_traces) == 3 * (5 + 8)
    assert [record.method for record in first.records] == [record.method for record in second.records]
    assert np.allclose(
        [record.nmse for record in first.records], [record.nmse for record in second.records]
    )
    assert np.all(np.isfinite([trace.information_score for trace in first.action_traces]))

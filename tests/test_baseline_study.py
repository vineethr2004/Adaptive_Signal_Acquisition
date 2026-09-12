import numpy as np

from adaptive_signal_acquisition.experiments import BaselineStudyConfig, run_baseline_study
from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.signals import SignalConfig


def test_baseline_study_is_paired_and_reproducible() -> None:
    config = BaselineStudyConfig(
        signal=SignalConfig(dimension=24, sparsity=2),
        candidate_count=48,
        budgets=(8, 12),
        trials_per_budget=4,
        noise_std=0.02,
        lasso=LassoConfig(lambda_value=0.03),
        seed=2026,
    )
    first = run_baseline_study(config)
    second = run_baseline_study(config)

    assert len(first.records) == 2 * 2 * 4
    assert len(first.summaries) == 2 * 2
    assert [record.method for record in first.records] == [record.method for record in second.records]
    assert np.allclose(
        [record.nmse for record in first.records], [record.nmse for record in second.records]
    )
    assert np.all(np.isfinite([record.nmse for record in first.records]))
    assert np.all(np.isfinite([record.support_f1 for record in first.records]))

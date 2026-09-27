import inspect

import numpy as np

from adaptive_signal_acquisition.adaptive_v2 import (
    BootstrapPolicyConfig,
    estimate_bootstrap_uncertainty,
    run_adaptive_v2_sensing,
    select_adaptive_v2_action,
)
from adaptive_signal_acquisition.measurements import create_candidate_dictionary
from adaptive_signal_acquisition.reconstruction import LassoConfig
from adaptive_signal_acquisition.sequential import GaussianBelief
from adaptive_signal_acquisition.signals import SignalConfig
from adaptive_signal_acquisition.stage4b import Stage4BConfig, run_stage4b_study


def small_policy() -> BootstrapPolicyConfig:
    return BootstrapPolicyConfig(
        noise_std=0.05,
        prior_variance=0.1,
        burn_in_steps=3,
        ensemble_size=6,
        gaussian_weight=0.25,
        ensemble_lasso=LassoConfig(
            lambda_value=0.03, max_iterations=100, tolerance=1e-5
        ),
    )


def test_adaptive_v2_is_reproducible_unique_and_observation_dependent() -> None:
    dictionary = create_candidate_dictionary(32, 12, np.random.default_rng(10))
    signal = np.zeros(12)
    signal[[2, 9]] = [1.5, -0.8]
    noise = np.array([0.04, -0.02, 0.01, 0.03, -0.01, 0.00, 0.02])

    first = run_adaptive_v2_sensing(
        dictionary, signal, noise, np.random.default_rng(101), small_policy()
    )
    second = run_adaptive_v2_sensing(
        dictionary, signal, noise, np.random.default_rng(101), small_policy()
    )
    assert np.array_equal(first.action_indices, second.action_indices)
    assert np.allclose(first.observations, first.sensing_matrix @ signal + noise)
    assert len(np.unique(first.action_indices)) == len(noise)
    assert first.phases[:3] == ("burn_in",) * 3
    assert first.phases[3:] == ("bootstrap",) * 4
    assert np.all(first.bootstrap_covariance_ranks[3:] > 0)
    assert np.any(first.selected_bootstrap_components[3:] > 0)

    belief = GaussianBelief.initial(12, 0.1)
    sensing_matrix = dictionary[:4]
    config = small_policy()
    zero_history = estimate_bootstrap_uncertainty(
        sensing_matrix,
        np.zeros(4),
        belief.covariance,
        config,
        np.random.default_rng(9),
    )
    nonzero_history = estimate_bootstrap_uncertainty(
        sensing_matrix,
        np.array([1.0, -0.3, 0.8, -0.2]),
        belief.covariance,
        config,
        np.random.default_rng(9),
    )
    assert not np.allclose(zero_history.covariance, nonzero_history.covariance)


def test_selector_cannot_receive_hidden_signal() -> None:
    assert "hidden_signal" not in inspect.signature(select_adaptive_v2_action).parameters


def test_stage4b_study_is_paired_and_reproducible() -> None:
    lasso = LassoConfig(lambda_value=0.03, max_iterations=150, tolerance=1e-5)
    config = Stage4BConfig(
        signal=SignalConfig(dimension=16, sparsity=2),
        candidate_count=32,
        budgets=(4, 7),
        trials_per_budget=2,
        noise_std=0.05,
        final_lasso=lasso,
        adaptive_v1_lasso=lasso,
        adaptive_v2=BootstrapPolicyConfig(
            noise_std=0.05,
            prior_variance=0.125,
            burn_in_steps=3,
            ensemble_size=5,
            gaussian_weight=0.25,
            ensemble_lasso=lasso,
        ),
        bootstrap_resamples=50,
        seed=405,
    )
    first = run_stage4b_study(config)
    second = run_stage4b_study(config)
    assert len(first.records) == 5 * 2 * 2
    assert len(first.summaries) == 5 * 2
    assert len(first.diagnostic_traces) == 2 * 7
    assert np.allclose(
        [row.nmse for row in first.records], [row.nmse for row in second.records]
    )

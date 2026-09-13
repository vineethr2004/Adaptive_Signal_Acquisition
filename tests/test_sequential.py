import numpy as np

from adaptive_signal_acquisition.measurements import create_candidate_dictionary
from adaptive_signal_acquisition.sequential import (
    GaussianBelief,
    InformationGuidedPolicyConfig,
    information_scores,
    run_information_guided_sensing,
    scoring_covariance,
    select_information_guided_action,
)


def test_information_guided_run_is_deterministic_and_uses_each_action_once() -> None:
    dictionary = create_candidate_dictionary(32, 12, np.random.default_rng(10))
    signal = np.zeros(12)
    signal[[2, 9]] = [1.5, -0.8]
    noise = np.array([0.04, -0.02, 0.01, 0.03, -0.01, 0.00])
    config = InformationGuidedPolicyConfig(noise_std=0.05, prior_variance=0.1)

    first = run_information_guided_sensing(dictionary, signal, noise, config)
    second = run_information_guided_sensing(dictionary, signal, noise, config)

    assert np.array_equal(first.action_indices, second.action_indices)
    assert np.allclose(first.observations, second.observations)
    assert len(np.unique(first.action_indices)) == len(noise)
    assert np.allclose(first.observations, first.sensing_matrix @ signal + noise)


def test_policy_score_changes_when_observation_history_changes() -> None:
    dictionary = create_candidate_dictionary(24, 8, np.random.default_rng(3))
    belief = GaussianBelief.initial(8, prior_variance=0.1)
    config = InformationGuidedPolicyConfig(noise_std=0.05, support_proxy_weight=1.0)
    zero_history = np.zeros(8)
    evidence_for_coordinate_zero = np.zeros(8)
    evidence_for_coordinate_zero[0] = 3.0

    baseline_scores = information_scores(
        dictionary, scoring_covariance(belief, zero_history, 1.0), config.noise_std**2
    )
    history_scores = information_scores(
        dictionary, scoring_covariance(belief, evidence_for_coordinate_zero, 1.0), config.noise_std**2
    )
    baseline_action, _ = select_information_guided_action(
        dictionary, np.array([], dtype=np.int64), belief, zero_history, config
    )
    history_action, _ = select_information_guided_action(
        dictionary, np.array([], dtype=np.int64), belief, evidence_for_coordinate_zero, config
    )

    assert not np.allclose(baseline_scores, history_scores)
    assert 0 <= baseline_action < len(dictionary)
    assert 0 <= history_action < len(dictionary)


def test_policy_api_has_no_hidden_signal_argument() -> None:
    """The selector only accepts the candidate actions and already-known state."""

    assert "hidden_signal" not in select_information_guided_action.__code__.co_varnames

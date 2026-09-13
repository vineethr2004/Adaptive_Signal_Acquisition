"""Sequential, information-guided measurement selection for Stage 3.

The policy in this module never receives the hidden signal ``x``.  It only
sees the actions it has already taken and their observed noisy values.  The
simulator owns ``x`` and uses it only after an action has been selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .reconstruction import LassoConfig, lasso_ista


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GaussianBelief:
    """A transparent Gaussian approximation to uncertainty about the signal."""

    mean: FloatArray
    covariance: FloatArray

    @classmethod
    def initial(cls, signal_dimension: int, prior_variance: float) -> "GaussianBelief":
        if signal_dimension <= 0:
            raise ValueError("signal_dimension must be positive")
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        return cls(
            mean=np.zeros(signal_dimension, dtype=np.float64),
            covariance=prior_variance * np.eye(signal_dimension, dtype=np.float64),
        )

    def updated(
        self, action: FloatArray, observation: float, noise_variance: float
    ) -> "GaussianBelief":
        """Condition the Gaussian approximation on one linear measurement."""

        if action.shape != self.mean.shape:
            raise ValueError("action and belief mean must have the same dimension")
        if noise_variance <= 0:
            raise ValueError("noise_variance must be positive for the information score")

        covariance_action = self.covariance @ action
        predictive_variance = float(action @ covariance_action + noise_variance)
        gain = covariance_action / predictive_variance
        mean = self.mean + gain * (observation - float(action @ self.mean))
        covariance = self.covariance - np.outer(gain, covariance_action)
        # Round-off can make a theoretically symmetric covariance slightly asymmetric.
        covariance = 0.5 * (covariance + covariance.T)
        return GaussianBelief(mean=mean, covariance=covariance)


@dataclass(frozen=True)
class InformationGuidedPolicyConfig:
    """Settings for the documented Stage 3 information-guided proxy policy.

    ``support_proxy_weight`` controls a data-dependent, positive-semidefinite
    addition ``weight * z_hat z_hat.T``.  ``z_hat`` is a provisional LASSO
    estimate based only on past actions and observations.  It is not an exact
    sparse posterior; it is a deliberately simple proxy that lets the next
    choice react to the observation history.
    """

    noise_std: float = 0.10
    prior_variance: float = 0.0625
    support_proxy_weight: float = 1.0
    provisional_lasso: LassoConfig = LassoConfig(max_iterations=1_000)

    def __post_init__(self) -> None:
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if self.support_proxy_weight < 0:
            raise ValueError("support_proxy_weight must be non-negative")


@dataclass(frozen=True)
class SequentialSensingRun:
    """Complete action/observation history from one adaptive acquisition run."""

    action_indices: IntArray
    observations: FloatArray
    selected_scores: FloatArray
    sensing_matrix: FloatArray
    final_belief: GaussianBelief


def scoring_covariance(
    belief: GaussianBelief, provisional_estimate: FloatArray, support_proxy_weight: float
) -> FloatArray:
    """Return the Gaussian covariance plus the documented history-based proxy."""

    if provisional_estimate.shape != belief.mean.shape:
        raise ValueError("provisional_estimate and belief must have the same dimension")
    return belief.covariance + support_proxy_weight * np.outer(
        provisional_estimate, provisional_estimate
    )


def information_scores(
    dictionary: FloatArray, covariance: FloatArray, noise_variance: float
) -> FloatArray:
    """Score actions by ``log(1 + a.T covariance a / noise_variance)``."""

    if dictionary.ndim != 2 or covariance.shape != (dictionary.shape[1], dictionary.shape[1]):
        raise ValueError("dictionary and covariance have incompatible shapes")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")
    action_variances = np.einsum("ij,jk,ik->i", dictionary, covariance, dictionary)
    return np.log1p(np.maximum(action_variances, 0.0) / noise_variance)


def select_information_guided_action(
    dictionary: FloatArray,
    used_action_indices: IntArray,
    belief: GaussianBelief,
    provisional_estimate: FloatArray,
    config: InformationGuidedPolicyConfig = InformationGuidedPolicyConfig(),
) -> tuple[int, float]:
    """Choose the highest-scoring unused action using only the current history."""

    covariance = scoring_covariance(belief, provisional_estimate, config.support_proxy_weight)
    scores = information_scores(dictionary, covariance, config.noise_std**2)
    scores[used_action_indices] = -np.inf
    action_index = int(np.argmax(scores))
    if not np.isfinite(scores[action_index]):
        raise ValueError("no unused candidate action remains")
    return action_index, float(scores[action_index])


def run_information_guided_sensing(
    dictionary: FloatArray,
    hidden_signal: FloatArray,
    noise: FloatArray,
    config: InformationGuidedPolicyConfig = InformationGuidedPolicyConfig(),
) -> SequentialSensingRun:
    """Acquire ``len(noise)`` measurements one at a time.

    The ordering is important: selection happens first, then the simulator
    evaluates the selected row against ``hidden_signal`` to create an
    observation.  Consequently, the selection code cannot leak the truth.
    """

    if dictionary.ndim != 2 or hidden_signal.shape != (dictionary.shape[1],):
        raise ValueError("dictionary and hidden_signal have incompatible shapes")
    if noise.ndim != 1 or not 0 < noise.size <= dictionary.shape[0]:
        raise ValueError("noise must contain between one and candidate_count values")

    belief = GaussianBelief.initial(dictionary.shape[1], config.prior_variance)
    used: list[int] = []
    observations: list[float] = []
    selected_scores: list[float] = []
    provisional = np.zeros(dictionary.shape[1], dtype=np.float64)

    for noise_value in noise:
        action_index, score = select_information_guided_action(
            dictionary,
            np.asarray(used, dtype=np.int64),
            belief,
            provisional,
            config,
        )
        action = dictionary[action_index]
        # The following line belongs to the simulated physical environment, not the policy.
        observation = float(action @ hidden_signal + noise_value)
        used.append(action_index)
        observations.append(observation)
        selected_scores.append(score)
        sensing_matrix = dictionary[np.asarray(used, dtype=np.int64)]
        observation_vector = np.asarray(observations, dtype=np.float64)
        belief = belief.updated(action, observation, config.noise_std**2)
        provisional = lasso_ista(sensing_matrix, observation_vector, config.provisional_lasso).coefficients

    action_indices = np.asarray(used, dtype=np.int64)
    return SequentialSensingRun(
        action_indices=action_indices,
        observations=np.asarray(observations, dtype=np.float64),
        selected_scores=np.asarray(selected_scores, dtype=np.float64),
        sensing_matrix=dictionary[action_indices].copy(),
        final_belief=belief,
    )

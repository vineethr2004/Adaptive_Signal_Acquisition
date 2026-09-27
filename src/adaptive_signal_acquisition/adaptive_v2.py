"""Observation-dependent uncertainty sensing for Stage 4B.

``adaptive_v2`` replaces the Stage 3 magnitude proxy with an empirical
covariance computed from a deterministic bootstrap ensemble of plausible
sparse reconstructions.  The policy only receives past actions and
observations.  The hidden signal is used by the simulator *after* selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .reconstruction import LassoConfig, soft_threshold
from .sequential import GaussianBelief, information_scores


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BootstrapPolicyConfig:
    """Configuration of the Stage 4B bootstrap-uncertainty policy.

    ``gaussian_weight`` is the shrinkage weight rho.  After burn-in, the
    score covariance is ``rho * Sigma_G + (1-rho) * Sigma_boot_scaled``.
    Trace matching makes rho control covariance *shape* rather than an
    arbitrary scale difference between the two estimators.
    """

    noise_std: float = 0.10
    prior_variance: float = 0.0625
    burn_in_steps: int = 8
    ensemble_size: int = 12
    perturbation_scale: float = 1.0
    gaussian_weight: float = 0.25
    ensemble_lasso: LassoConfig = LassoConfig(
        lambda_value=0.08, max_iterations=500, tolerance=1e-5
    )

    def __post_init__(self) -> None:
        if self.noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if self.burn_in_steps < 1:
            raise ValueError("burn_in_steps must be at least one")
        if self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two")
        if self.perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be positive")
        if not 0.0 <= self.gaussian_weight <= 1.0:
            raise ValueError("gaussian_weight must lie in [0, 1]")


@dataclass(frozen=True)
class BootstrapUncertainty:
    """Bootstrap ensemble and covariance used at one sequential step."""

    estimates: FloatArray
    mean: FloatArray
    covariance: FloatArray
    scaled_covariance: FloatArray
    covariance_trace: float
    covariance_rank: int


@dataclass(frozen=True)
class AdaptiveV2Run:
    """Complete auditable history from one Stage 4B sensing run."""

    action_indices: IntArray
    observations: FloatArray
    phases: tuple[str, ...]
    selected_scores: FloatArray
    selected_gaussian_components: FloatArray
    selected_bootstrap_components: FloatArray
    bootstrap_covariance_traces: FloatArray
    bootstrap_covariance_ranks: IntArray
    max_previous_action_correlations: FloatArray
    sensing_matrix_smallest_singular_values: FloatArray
    sensing_matrix_condition_numbers: FloatArray
    provisional_estimates: FloatArray
    sensing_matrix: FloatArray
    final_belief: GaussianBelief


def _lasso_ensemble_ista(
    sensing_matrix: FloatArray,
    observation_ensemble: FloatArray,
    config: LassoConfig,
) -> FloatArray:
    """Solve many LASSO problems together; columns are bootstrap replicates."""

    if sensing_matrix.ndim != 2 or observation_ensemble.ndim != 2:
        raise ValueError("sensing_matrix and observation_ensemble must be matrices")
    measurement_count, signal_dimension = sensing_matrix.shape
    if observation_ensemble.shape[0] != measurement_count:
        raise ValueError("each bootstrap observation must match the sensing rows")
    lipschitz = float(np.linalg.norm(sensing_matrix, ord=2) ** 2)
    if lipschitz == 0.0:
        raise ValueError("sensing_matrix must contain a non-zero row")

    step_size = 1.0 / lipschitz
    coefficients = np.zeros(
        (signal_dimension, observation_ensemble.shape[1]), dtype=np.float64
    )
    for _ in range(config.max_iterations):
        residual = observation_ensemble - sensing_matrix @ coefficients
        gradient_step = coefficients + step_size * (sensing_matrix.T @ residual)
        updated = soft_threshold(gradient_step, step_size * config.lambda_value)
        column_changes = np.linalg.norm(updated - coefficients, axis=0)
        column_scales = np.maximum(1.0, np.linalg.norm(coefficients, axis=0))
        coefficients = updated
        if np.all(column_changes <= config.tolerance * column_scales):
            break
    return coefficients.T


def estimate_bootstrap_uncertainty(
    sensing_matrix: FloatArray,
    observations: FloatArray,
    gaussian_covariance: FloatArray,
    config: BootstrapPolicyConfig,
    rng: np.random.Generator,
) -> BootstrapUncertainty:
    """Estimate observation-dependent signal uncertainty by parametric bootstrap.

    The same observed vector is perturbed ``R`` times by fresh noise with
    standard deviation ``perturbation_scale * noise_std``.  Reconstructing
    every perturbed data set produces an ensemble whose covariance describes
    directions in which plausible sparse solutions disagree.
    """

    if sensing_matrix.ndim != 2 or observations.shape != (sensing_matrix.shape[0],):
        raise ValueError("observations must match the rows of sensing_matrix")
    signal_dimension = sensing_matrix.shape[1]
    if gaussian_covariance.shape != (signal_dimension, signal_dimension):
        raise ValueError("gaussian_covariance has an incompatible shape")

    perturbations = rng.normal(
        0.0,
        config.perturbation_scale * config.noise_std,
        size=(observations.size, config.ensemble_size),
    )
    bootstrap_observations = observations[:, None] + perturbations
    estimates = _lasso_ensemble_ista(
        sensing_matrix, bootstrap_observations, config.ensemble_lasso
    )
    mean = np.mean(estimates, axis=0)
    centered = estimates - mean
    covariance = centered.T @ centered / (config.ensemble_size - 1)
    covariance = 0.5 * (covariance + covariance.T)
    covariance_trace = float(np.trace(covariance))
    covariance_rank = int(np.linalg.matrix_rank(covariance, tol=1e-10))

    gaussian_trace = float(np.trace(gaussian_covariance))
    if covariance_trace <= np.finfo(np.float64).eps:
        scaled_covariance = np.zeros_like(covariance)
    else:
        scaled_covariance = covariance * (gaussian_trace / covariance_trace)
    return BootstrapUncertainty(
        estimates=estimates,
        mean=mean,
        covariance=covariance,
        scaled_covariance=scaled_covariance,
        covariance_trace=covariance_trace,
        covariance_rank=covariance_rank,
    )


def select_adaptive_v2_action(
    dictionary: FloatArray,
    used_action_indices: IntArray,
    belief: GaussianBelief,
    sensing_matrix: FloatArray,
    observations: FloatArray,
    config: BootstrapPolicyConfig,
    rng: np.random.Generator,
) -> tuple[int, float, float, float, BootstrapUncertainty | None]:
    """Select an unused action using only the currently available history."""

    step = len(used_action_indices)
    if step < config.burn_in_steps:
        gaussian_covariance = belief.covariance
        bootstrap = None
        gaussian_weight = 1.0
        bootstrap_weight = 0.0
    else:
        bootstrap = estimate_bootstrap_uncertainty(
            sensing_matrix, observations, belief.covariance, config, rng
        )
        gaussian_weight = config.gaussian_weight
        bootstrap_weight = 1.0 - config.gaussian_weight
        if bootstrap.covariance_trace <= np.finfo(np.float64).eps:
            gaussian_weight = 1.0
            bootstrap_weight = 0.0
        gaussian_covariance = (
            gaussian_weight * belief.covariance
            + bootstrap_weight * bootstrap.scaled_covariance
        )

    scores = information_scores(dictionary, gaussian_covariance, config.noise_std**2)
    available_scores = scores.copy()
    available_scores[used_action_indices] = -np.inf
    action_index = int(np.argmax(available_scores))
    if not np.isfinite(available_scores[action_index]):
        raise ValueError("no unused candidate action remains")

    action = dictionary[action_index]
    gaussian_component = gaussian_weight * float(action @ belief.covariance @ action)
    bootstrap_component = 0.0
    if bootstrap is not None:
        bootstrap_component = bootstrap_weight * float(
            action @ bootstrap.scaled_covariance @ action
        )
    return (
        action_index,
        float(available_scores[action_index]),
        gaussian_component,
        bootstrap_component,
        bootstrap,
    )


def run_adaptive_v2_sensing(
    dictionary: FloatArray,
    hidden_signal: FloatArray,
    noise: FloatArray,
    rng: np.random.Generator,
    config: BootstrapPolicyConfig = BootstrapPolicyConfig(),
) -> AdaptiveV2Run:
    """Acquire measurements sequentially without exposing truth to the policy."""

    if dictionary.ndim != 2 or hidden_signal.shape != (dictionary.shape[1],):
        raise ValueError("dictionary and hidden_signal have incompatible shapes")
    if noise.ndim != 1 or not 0 < noise.size <= dictionary.shape[0]:
        raise ValueError("noise must contain between one and candidate_count values")

    belief = GaussianBelief.initial(dictionary.shape[1], config.prior_variance)
    used: list[int] = []
    observations: list[float] = []
    phases: list[str] = []
    selected_scores: list[float] = []
    gaussian_components: list[float] = []
    bootstrap_components: list[float] = []
    bootstrap_traces: list[float] = []
    bootstrap_ranks: list[int] = []
    correlations: list[float] = []
    smallest_singular_values: list[float] = []
    condition_numbers: list[float] = []
    provisional_estimates: list[FloatArray] = []

    for noise_value in noise:
        used_array = np.asarray(used, dtype=np.int64)
        sensing_matrix = dictionary[used_array] if used else np.empty((0, dictionary.shape[1]))
        observation_vector = np.asarray(observations, dtype=np.float64)
        action_index, score, gaussian, bootstrap_component, bootstrap = (
            select_adaptive_v2_action(
                dictionary,
                used_array,
                belief,
                sensing_matrix,
                observation_vector,
                config,
                rng,
            )
        )
        action = dictionary[action_index]
        # Simulated environment: truth is accessed only after the policy selects a row.
        observation = float(action @ hidden_signal + noise_value)
        used.append(action_index)
        observations.append(observation)
        phases.append("burn_in" if bootstrap is None else "bootstrap")
        selected_scores.append(score)
        gaussian_components.append(gaussian)
        bootstrap_components.append(bootstrap_component)
        bootstrap_traces.append(0.0 if bootstrap is None else bootstrap.covariance_trace)
        bootstrap_ranks.append(0 if bootstrap is None else bootstrap.covariance_rank)

        updated_sensing_matrix = dictionary[np.asarray(used, dtype=np.int64)]
        if len(used) == 1:
            correlations.append(0.0)
        else:
            correlations.append(
                float(np.max(np.abs(updated_sensing_matrix[:-1] @ action)))
            )
        singular_values = np.linalg.svd(updated_sensing_matrix, compute_uv=False)
        smallest_singular_values.append(float(singular_values[-1]))
        condition_numbers.append(
            float("inf")
            if singular_values[-1] <= np.finfo(np.float64).eps
            else float(singular_values[0] / singular_values[-1])
        )

        belief = belief.updated(action, observation, config.noise_std**2)
        # Record a like-for-like diagnostic after the new observation, as in
        # Stage 4A.  The pre-observation bootstrap mean remains part of the
        # selection calculation but is not mislabeled as a post-step estimate.
        current_observations = np.asarray(observations, dtype=np.float64)[:, None]
        provisional = _lasso_ensemble_ista(
            updated_sensing_matrix, current_observations, config.ensemble_lasso
        )[0]
        provisional_estimates.append(provisional.copy())

    indices = np.asarray(used, dtype=np.int64)
    return AdaptiveV2Run(
        action_indices=indices,
        observations=np.asarray(observations, dtype=np.float64),
        phases=tuple(phases),
        selected_scores=np.asarray(selected_scores, dtype=np.float64),
        selected_gaussian_components=np.asarray(gaussian_components, dtype=np.float64),
        selected_bootstrap_components=np.asarray(bootstrap_components, dtype=np.float64),
        bootstrap_covariance_traces=np.asarray(bootstrap_traces, dtype=np.float64),
        bootstrap_covariance_ranks=np.asarray(bootstrap_ranks, dtype=np.int64),
        max_previous_action_correlations=np.asarray(correlations, dtype=np.float64),
        sensing_matrix_smallest_singular_values=np.asarray(
            smallest_singular_values, dtype=np.float64
        ),
        sensing_matrix_condition_numbers=np.asarray(condition_numbers, dtype=np.float64),
        provisional_estimates=np.asarray(provisional_estimates, dtype=np.float64),
        sensing_matrix=dictionary[indices].copy(),
        final_belief=belief,
    )

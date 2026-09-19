"""Stage 4A controlled diagnostics for the first adaptive sensing policy.

This module freezes Stage 3 as ``adaptive_v1`` and varies only the weight of
its provisional-LASSO proxy.  It records enough intermediate quantities to
answer *why* an action was selected and whether the resulting sensing matrix
is becoming redundant or poorly conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .baselines import fixed_action_indices, make_fixed_action_order
from .experiments import BaselineStudyConfig, TrialRecord
from .measurements import (
    create_candidate_dictionary,
    sensing_matrix_from_indices,
    simulate_measurements,
)
from .metrics import normalized_mean_squared_error, support_f1_score
from .reconstruction import LassoConfig, lasso_ista
from .sequential import InformationGuidedPolicyConfig, run_information_guided_sensing
from .signals import SignalConfig, generate_sparse_signal


@dataclass(frozen=True)
class Stage4AConfig:
    """Configuration for the focused Gaussian/LASSO-proxy ablation."""

    signal: SignalConfig = SignalConfig()
    candidate_count: int = 128
    budgets: tuple[int, ...] = (8, 12, 16, 20, 24, 32)
    trials_per_budget: int = 100
    noise_std: float = 0.10
    lasso: LassoConfig = LassoConfig()
    provisional_lasso: LassoConfig = LassoConfig()
    prior_variance: float = 0.0625
    gamma_values: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
    bootstrap_resamples: int = 2_000
    seed: int = 20_260_919

    def __post_init__(self) -> None:
        BaselineStudyConfig(
            signal=self.signal,
            candidate_count=self.candidate_count,
            budgets=self.budgets,
            trials_per_budget=self.trials_per_budget,
            noise_std=self.noise_std,
            lasso=self.lasso,
            seed=self.seed,
        )
        if self.noise_std <= 0:
            raise ValueError("Stage 4A requires positive noise_std")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if not self.gamma_values or any(gamma < 0 for gamma in self.gamma_values):
            raise ValueError("gamma_values must contain non-negative values")
        if len(set(self.gamma_values)) != len(self.gamma_values):
            raise ValueError("gamma_values must be unique")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True)
class Stage4ATraceRecord:
    """One adaptive decision and its diagnostic quantities."""

    method: str
    gamma: float
    trial: int
    step: int
    action_index: int
    observation: float
    total_information_score: float
    gaussian_variance_component: float
    lasso_proxy_component: float
    lasso_component_fraction: float
    max_previous_action_correlation: float
    smallest_singular_value: float
    sensing_matrix_condition_number: float
    provisional_nmse: float
    provisional_support_f1: float


@dataclass(frozen=True)
class Stage4ASummaryRecord:
    """Aggregate reconstruction result and paired uncertainty interval."""

    method: str
    budget: int
    mean_nmse: float
    median_nmse: float
    mean_support_f1: float
    mean_reconstruction_ms: float
    convergence_rate: float
    mean_nmse_difference_vs_random: float
    difference_ci_lower_95: float
    difference_ci_upper_95: float


@dataclass(frozen=True)
class Stage4AResult:
    """All records produced by the Stage 4A diagnostic experiment."""

    config: Stage4AConfig
    records: tuple[TrialRecord, ...]
    summaries: tuple[Stage4ASummaryRecord, ...]
    diagnostic_traces: tuple[Stage4ATraceRecord, ...]


def gamma_method_name(gamma: float) -> str:
    """Create a stable method label without implying a new learned policy."""

    return f"adaptive_v1_gamma_{gamma:g}"


def _append_reconstruction(
    records: list[TrialRecord],
    method: str,
    budget: int,
    trial: int,
    sensing_matrix: np.ndarray,
    observations: np.ndarray,
    hidden_signal: np.ndarray,
    true_support: np.ndarray,
    lasso: LassoConfig,
) -> None:
    start = perf_counter()
    reconstruction = lasso_ista(sensing_matrix, observations, lasso)
    elapsed_ms = (perf_counter() - start) * 1_000.0
    records.append(
        TrialRecord(
            method=method,
            budget=budget,
            trial=trial,
            nmse=normalized_mean_squared_error(hidden_signal, reconstruction.coefficients),
            support_f1=support_f1_score(true_support, reconstruction.coefficients),
            reconstruction_ms=elapsed_ms,
            converged=reconstruction.converged,
        )
    )


def _bootstrap_mean_interval(
    differences: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a paired mean."""

    sample_indices = rng.integers(
        0, differences.size, size=(resamples, differences.size), endpoint=False
    )
    bootstrap_means = np.mean(differences[sample_indices], axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def _summarize(
    records: list[TrialRecord], config: Stage4AConfig, method_order: tuple[str, ...]
) -> tuple[Stage4ASummaryRecord, ...]:
    bootstrap_rng = np.random.default_rng(config.seed + 4_000)
    summaries: list[Stage4ASummaryRecord] = []
    for method in method_order:
        for budget in config.budgets:
            group = sorted(
                (
                    record
                    for record in records
                    if record.method == method and record.budget == budget
                ),
                key=lambda record: record.trial,
            )
            random_group = sorted(
                (
                    record
                    for record in records
                    if record.method == "random" and record.budget == budget
                ),
                key=lambda record: record.trial,
            )
            if len(group) != config.trials_per_budget or len(random_group) != len(group):
                raise RuntimeError("incomplete paired records during Stage 4A summarization")
            nmse = np.asarray([record.nmse for record in group])
            random_nmse = np.asarray([record.nmse for record in random_group])
            differences = nmse - random_nmse
            lower, upper = _bootstrap_mean_interval(
                differences, config.bootstrap_resamples, bootstrap_rng
            )
            summaries.append(
                Stage4ASummaryRecord(
                    method=method,
                    budget=budget,
                    mean_nmse=float(np.mean(nmse)),
                    median_nmse=float(np.median(nmse)),
                    mean_support_f1=float(np.mean([record.support_f1 for record in group])),
                    mean_reconstruction_ms=float(
                        np.mean([record.reconstruction_ms for record in group])
                    ),
                    convergence_rate=float(
                        np.mean([record.converged for record in group])
                    ),
                    mean_nmse_difference_vs_random=float(np.mean(differences)),
                    difference_ci_lower_95=lower,
                    difference_ci_upper_95=upper,
                )
            )
    return tuple(summaries)


def run_stage4a_diagnostics(config: Stage4AConfig = Stage4AConfig()) -> Stage4AResult:
    """Run paired gamma ablations and collect action-level diagnostics.

    Every adaptive gamma value is run once to the maximum budget per trial.
    Smaller-budget results reuse exact prefixes of that run.  This is both
    faster and equivalent to stopping the deterministic sequential policy at
    an earlier budget.
    """

    dictionary_seed, fixed_seed, trial_seed = np.random.SeedSequence(config.seed).spawn(3)
    dictionary = create_candidate_dictionary(
        config.candidate_count,
        config.signal.dimension,
        np.random.default_rng(dictionary_seed),
    )
    fixed_order = make_fixed_action_order(
        config.candidate_count, np.random.default_rng(fixed_seed)
    )
    trial_seeds = trial_seed.spawn(config.trials_per_budget)
    maximum_budget = max(config.budgets)
    records: list[TrialRecord] = []
    traces: list[Stage4ATraceRecord] = []

    for trial, seed_sequence in enumerate(trial_seeds):
        signal_seed, noise_seed, random_seed = seed_sequence.spawn(3)
        hidden_signal, true_support = generate_sparse_signal(
            config.signal, np.random.default_rng(signal_seed)
        )
        paired_noise = np.random.default_rng(noise_seed).normal(
            0.0, config.noise_std, size=maximum_budget
        )
        random_order = np.random.default_rng(random_seed).permutation(
            config.candidate_count
        ).astype(np.int64)

        for budget in config.budgets:
            for method, action_indices in (
                ("fixed", fixed_action_indices(fixed_order, budget)),
                ("random", random_order[:budget]),
            ):
                sensing_matrix = sensing_matrix_from_indices(dictionary, action_indices)
                _, observations = simulate_measurements(
                    sensing_matrix, hidden_signal, paired_noise[:budget]
                )
                _append_reconstruction(
                    records,
                    method,
                    budget,
                    trial,
                    sensing_matrix,
                    observations,
                    hidden_signal,
                    true_support,
                    config.lasso,
                )

        for gamma in config.gamma_values:
            method = gamma_method_name(gamma)
            policy = InformationGuidedPolicyConfig(
                noise_std=config.noise_std,
                prior_variance=config.prior_variance,
                support_proxy_weight=gamma,
                provisional_lasso=config.provisional_lasso,
            )
            sequential_run = run_information_guided_sensing(
                dictionary, hidden_signal, paired_noise, policy
            )

            for budget in config.budgets:
                _append_reconstruction(
                    records,
                    method,
                    budget,
                    trial,
                    sequential_run.sensing_matrix[:budget],
                    sequential_run.observations[:budget],
                    hidden_signal,
                    true_support,
                    config.lasso,
                )

            for step in range(maximum_budget):
                gaussian = float(sequential_run.selected_gaussian_components[step])
                lasso_proxy = float(sequential_run.selected_lasso_components[step])
                total_component = gaussian + lasso_proxy
                lasso_fraction = lasso_proxy / total_component if total_component > 0 else 0.0
                provisional = sequential_run.provisional_estimates[step]
                traces.append(
                    Stage4ATraceRecord(
                        method=method,
                        gamma=gamma,
                        trial=trial,
                        step=step + 1,
                        action_index=int(sequential_run.action_indices[step]),
                        observation=float(sequential_run.observations[step]),
                        total_information_score=float(sequential_run.selected_scores[step]),
                        gaussian_variance_component=gaussian,
                        lasso_proxy_component=lasso_proxy,
                        lasso_component_fraction=float(lasso_fraction),
                        max_previous_action_correlation=float(
                            sequential_run.max_previous_action_correlations[step]
                        ),
                        smallest_singular_value=float(
                            sequential_run.sensing_matrix_smallest_singular_values[step]
                        ),
                        sensing_matrix_condition_number=float(
                            sequential_run.sensing_matrix_condition_numbers[step]
                        ),
                        provisional_nmse=normalized_mean_squared_error(
                            hidden_signal, provisional
                        ),
                        provisional_support_f1=support_f1_score(
                            true_support, provisional
                        ),
                    )
                )

    method_order = (
        "fixed",
        "random",
        *(gamma_method_name(gamma) for gamma in config.gamma_values),
    )
    return Stage4AResult(
        config=config,
        records=tuple(records),
        summaries=_summarize(records, config, method_order),
        diagnostic_traces=tuple(traces),
    )

"""Stage 4B evaluation of observation-dependent bootstrap uncertainty sensing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .adaptive_v2 import BootstrapPolicyConfig, run_adaptive_v2_sensing
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
class Stage4BConfig:
    """Frozen paired protocol for comparing adaptive_v2 with prior methods."""

    signal: SignalConfig = SignalConfig()
    candidate_count: int = 128
    budgets: tuple[int, ...] = (8, 12, 16, 20, 24, 32)
    trials_per_budget: int = 100
    noise_std: float = 0.10
    final_lasso: LassoConfig = LassoConfig()
    adaptive_v1_lasso: LassoConfig = LassoConfig()
    adaptive_v2: BootstrapPolicyConfig = BootstrapPolicyConfig()
    bootstrap_resamples: int = 2_000
    seed: int = 20_260_927

    def __post_init__(self) -> None:
        BaselineStudyConfig(
            signal=self.signal,
            candidate_count=self.candidate_count,
            budgets=self.budgets,
            trials_per_budget=self.trials_per_budget,
            noise_std=self.noise_std,
            lasso=self.final_lasso,
            seed=self.seed,
        )
        if not np.isclose(self.noise_std, self.adaptive_v2.noise_std):
            raise ValueError("adaptive_v2.noise_std must equal experiment noise_std")
        if self.adaptive_v2.burn_in_steps > max(self.budgets):
            raise ValueError("burn_in_steps cannot exceed the maximum budget")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True)
class Stage4BSummaryRecord:
    """Aggregate result and paired NMSE interval against random sensing."""

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
class Stage4BTraceRecord:
    """One adaptive_v2 choice and the uncertainty evidence behind it."""

    trial: int
    step: int
    phase: str
    action_index: int
    observation: float
    total_information_score: float
    gaussian_variance_component: float
    bootstrap_variance_component: float
    bootstrap_component_fraction: float
    bootstrap_covariance_trace: float
    bootstrap_covariance_rank: int
    max_previous_action_correlation: float
    smallest_singular_value: float
    sensing_matrix_condition_number: float
    provisional_nmse: float
    provisional_support_f1: float


@dataclass(frozen=True)
class Stage4BResult:
    """Trial, summary, and decision-level evidence from Stage 4B."""

    config: Stage4BConfig
    records: tuple[TrialRecord, ...]
    summaries: tuple[Stage4BSummaryRecord, ...]
    diagnostic_traces: tuple[Stage4BTraceRecord, ...]


METHOD_ORDER = (
    "fixed",
    "random",
    "gaussian_only",
    "adaptive_v1_gamma_1",
    "adaptive_v2_bootstrap",
)


def _append_reconstruction(
    records: list[TrialRecord],
    method: str,
    budget: int,
    trial: int,
    sensing_matrix: np.ndarray,
    observations: np.ndarray,
    hidden_signal: np.ndarray,
    true_support: np.ndarray,
    config: LassoConfig,
) -> None:
    start = perf_counter()
    result = lasso_ista(sensing_matrix, observations, config)
    elapsed_ms = (perf_counter() - start) * 1_000.0
    records.append(
        TrialRecord(
            method=method,
            budget=budget,
            trial=trial,
            nmse=normalized_mean_squared_error(hidden_signal, result.coefficients),
            support_f1=support_f1_score(true_support, result.coefficients),
            reconstruction_ms=elapsed_ms,
            converged=result.converged,
        )
    )


def _paired_interval(
    differences: np.ndarray, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    indices = rng.integers(
        0, differences.size, size=(resamples, differences.size), endpoint=False
    )
    means = np.mean(differences[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _summarize(
    records: list[TrialRecord], config: Stage4BConfig
) -> tuple[Stage4BSummaryRecord, ...]:
    rng = np.random.default_rng(config.seed + 40_000)
    summaries: list[Stage4BSummaryRecord] = []
    for method in METHOD_ORDER:
        for budget in config.budgets:
            group = sorted(
                (r for r in records if r.method == method and r.budget == budget),
                key=lambda r: r.trial,
            )
            random_group = sorted(
                (r for r in records if r.method == "random" and r.budget == budget),
                key=lambda r: r.trial,
            )
            if len(group) != config.trials_per_budget or len(random_group) != len(group):
                raise RuntimeError("incomplete paired Stage 4B records")
            nmse = np.asarray([r.nmse for r in group])
            random_nmse = np.asarray([r.nmse for r in random_group])
            differences = nmse - random_nmse
            lower, upper = _paired_interval(
                differences, config.bootstrap_resamples, rng
            )
            summaries.append(
                Stage4BSummaryRecord(
                    method=method,
                    budget=budget,
                    mean_nmse=float(np.mean(nmse)),
                    median_nmse=float(np.median(nmse)),
                    mean_support_f1=float(np.mean([r.support_f1 for r in group])),
                    mean_reconstruction_ms=float(
                        np.mean([r.reconstruction_ms for r in group])
                    ),
                    convergence_rate=float(np.mean([r.converged for r in group])),
                    mean_nmse_difference_vs_random=float(np.mean(differences)),
                    difference_ci_lower_95=lower,
                    difference_ci_upper_95=upper,
                )
            )
    return tuple(summaries)


def run_stage4b_study(config: Stage4BConfig = Stage4BConfig()) -> Stage4BResult:
    """Run all methods once to maximum budget and reuse exact prefixes."""

    dictionary_seed, fixed_seed, trials_seed = np.random.SeedSequence(config.seed).spawn(3)
    dictionary = create_candidate_dictionary(
        config.candidate_count,
        config.signal.dimension,
        np.random.default_rng(dictionary_seed),
    )
    fixed_order = make_fixed_action_order(
        config.candidate_count, np.random.default_rng(fixed_seed)
    )
    trial_seeds = trials_seed.spawn(config.trials_per_budget)
    maximum_budget = max(config.budgets)
    records: list[TrialRecord] = []
    traces: list[Stage4BTraceRecord] = []

    for trial, trial_seed in enumerate(trial_seeds):
        signal_seed, noise_seed, random_seed, policy_seed = trial_seed.spawn(4)
        hidden_signal, true_support = generate_sparse_signal(
            config.signal, np.random.default_rng(signal_seed)
        )
        noise = np.random.default_rng(noise_seed).normal(
            0.0, config.noise_std, size=maximum_budget
        )
        random_order = np.random.default_rng(random_seed).permutation(
            config.candidate_count
        ).astype(np.int64)

        for budget in config.budgets:
            for method, indices in (
                ("fixed", fixed_action_indices(fixed_order, budget)),
                ("random", random_order[:budget]),
            ):
                sensing_matrix = sensing_matrix_from_indices(dictionary, indices)
                _, observations = simulate_measurements(
                    sensing_matrix, hidden_signal, noise[:budget]
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
                    config.final_lasso,
                )

        sequential_runs = {
            "gaussian_only": run_information_guided_sensing(
                dictionary,
                hidden_signal,
                noise,
                InformationGuidedPolicyConfig(
                    noise_std=config.noise_std,
                    prior_variance=config.adaptive_v2.prior_variance,
                    support_proxy_weight=0.0,
                    provisional_lasso=config.adaptive_v1_lasso,
                ),
            ),
            "adaptive_v1_gamma_1": run_information_guided_sensing(
                dictionary,
                hidden_signal,
                noise,
                InformationGuidedPolicyConfig(
                    noise_std=config.noise_std,
                    prior_variance=config.adaptive_v2.prior_variance,
                    support_proxy_weight=1.0,
                    provisional_lasso=config.adaptive_v1_lasso,
                ),
            ),
        }
        adaptive_v2_run = run_adaptive_v2_sensing(
            dictionary,
            hidden_signal,
            noise,
            np.random.default_rng(policy_seed),
            config.adaptive_v2,
        )

        for budget in config.budgets:
            for method, run in sequential_runs.items():
                _append_reconstruction(
                    records,
                    method,
                    budget,
                    trial,
                    run.sensing_matrix[:budget],
                    run.observations[:budget],
                    hidden_signal,
                    true_support,
                    config.final_lasso,
                )
            _append_reconstruction(
                records,
                "adaptive_v2_bootstrap",
                budget,
                trial,
                adaptive_v2_run.sensing_matrix[:budget],
                adaptive_v2_run.observations[:budget],
                hidden_signal,
                true_support,
                config.final_lasso,
            )

        for index in range(maximum_budget):
            gaussian = float(adaptive_v2_run.selected_gaussian_components[index])
            bootstrap = float(adaptive_v2_run.selected_bootstrap_components[index])
            total = gaussian + bootstrap
            provisional = adaptive_v2_run.provisional_estimates[index]
            traces.append(
                Stage4BTraceRecord(
                    trial=trial,
                    step=index + 1,
                    phase=adaptive_v2_run.phases[index],
                    action_index=int(adaptive_v2_run.action_indices[index]),
                    observation=float(adaptive_v2_run.observations[index]),
                    total_information_score=float(adaptive_v2_run.selected_scores[index]),
                    gaussian_variance_component=gaussian,
                    bootstrap_variance_component=bootstrap,
                    bootstrap_component_fraction=bootstrap / total if total > 0 else 0.0,
                    bootstrap_covariance_trace=float(
                        adaptive_v2_run.bootstrap_covariance_traces[index]
                    ),
                    bootstrap_covariance_rank=int(
                        adaptive_v2_run.bootstrap_covariance_ranks[index]
                    ),
                    max_previous_action_correlation=float(
                        adaptive_v2_run.max_previous_action_correlations[index]
                    ),
                    smallest_singular_value=float(
                        adaptive_v2_run.sensing_matrix_smallest_singular_values[index]
                    ),
                    sensing_matrix_condition_number=float(
                        adaptive_v2_run.sensing_matrix_condition_numbers[index]
                    ),
                    provisional_nmse=normalized_mean_squared_error(
                        hidden_signal, provisional
                    ),
                    provisional_support_f1=support_f1_score(
                        true_support, provisional
                    ),
                )
            )

    return Stage4BResult(
        config=config,
        records=tuple(records),
        summaries=_summarize(records, config),
        diagnostic_traces=tuple(traces),
    )

"""Paired, reproducible experiments for fixed, random, and adaptive sensing."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .baselines import fixed_action_indices, make_fixed_action_order, random_action_indices
from .measurements import (
    create_candidate_dictionary,
    sensing_matrix_from_indices,
    simulate_measurements,
)
from .metrics import normalized_mean_squared_error, support_f1_score
from .reconstruction import LassoConfig, lasso_ista
from .sequential import (
    InformationGuidedPolicyConfig,
    SequentialSensingRun,
    run_information_guided_sensing,
)
from .signals import SignalConfig, generate_sparse_signal


@dataclass(frozen=True)
class BaselineStudyConfig:
    """Frozen protocol for a fair fixed-versus-random Stage 2 comparison."""

    signal: SignalConfig = SignalConfig()
    candidate_count: int = 128
    budgets: tuple[int, ...] = (8, 12, 16, 20, 24, 32)
    trials_per_budget: int = 100
    noise_std: float = 0.10
    lasso: LassoConfig = LassoConfig()
    seed: int = 20_260_912

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not self.budgets or any(budget <= 0 for budget in self.budgets):
            raise ValueError("budgets must contain positive measurement counts")
        if any(budget > self.candidate_count for budget in self.budgets):
            raise ValueError("every budget must fit in the candidate dictionary")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be strictly increasing and unique")
        if self.trials_per_budget <= 0:
            raise ValueError("trials_per_budget must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")


@dataclass(frozen=True)
class TrialRecord:
    """One reconstruction result from one method, budget, and paired trial."""

    method: str
    budget: int
    trial: int
    nmse: float
    support_f1: float
    reconstruction_ms: float
    converged: bool


@dataclass(frozen=True)
class SummaryRecord:
    """Aggregate statistics for a method at one measurement budget."""

    method: str
    budget: int
    mean_nmse: float
    median_nmse: float
    mean_support_f1: float
    mean_reconstruction_ms: float
    convergence_rate: float


@dataclass(frozen=True)
class BaselineStudyResult:
    """All trial-level and aggregated results from a deterministic study."""

    config: BaselineStudyConfig
    records: tuple[TrialRecord, ...]
    summaries: tuple[SummaryRecord, ...]


def _summarize(records: list[TrialRecord], methods: tuple[str, ...]) -> tuple[SummaryRecord, ...]:
    summaries: list[SummaryRecord] = []
    for method in methods:
        for budget in sorted({record.budget for record in records}):
            group = [record for record in records if record.method == method and record.budget == budget]
            nmse = np.array([record.nmse for record in group])
            f1 = np.array([record.support_f1 for record in group])
            runtimes = np.array([record.reconstruction_ms for record in group])
            convergence = np.array([record.converged for record in group], dtype=np.float64)
            summaries.append(
                SummaryRecord(
                    method=method,
                    budget=budget,
                    mean_nmse=float(np.mean(nmse)),
                    median_nmse=float(np.median(nmse)),
                    mean_support_f1=float(np.mean(f1)),
                    mean_reconstruction_ms=float(np.mean(runtimes)),
                    convergence_rate=float(np.mean(convergence)),
                )
            )
    return tuple(summaries)


def run_baseline_study(config: BaselineStudyConfig = BaselineStudyConfig()) -> BaselineStudyResult:
    """Run fixed and random non-adaptive sensing on paired signals and noise.

    Within each trial and budget, both methods receive the same hidden signal
    and the same pre-drawn Gaussian noise vector.  They differ only in the
    rows selected for A.  Trials are independent replications; measurements
    are never pooled between trials.
    """

    dictionary_seed, fixed_seed, trial_seed = np.random.SeedSequence(config.seed).spawn(3)
    dictionary_rng = np.random.default_rng(dictionary_seed)
    fixed_rng = np.random.default_rng(fixed_seed)
    candidate_dictionary = create_candidate_dictionary(
        config.candidate_count, config.signal.dimension, dictionary_rng
    )
    fixed_order = make_fixed_action_order(config.candidate_count, fixed_rng)
    trial_seeds = trial_seed.spawn(config.trials_per_budget)
    maximum_budget = max(config.budgets)
    records: list[TrialRecord] = []

    for trial_index, seed_sequence in enumerate(trial_seeds):
        signal_seed, noise_seed, random_seed = seed_sequence.spawn(3)
        signal_rng = np.random.default_rng(signal_seed)
        noise_rng = np.random.default_rng(noise_seed)
        random_rng = np.random.default_rng(random_seed)
        x, true_support = generate_sparse_signal(config.signal, signal_rng)
        paired_noise = noise_rng.normal(0.0, config.noise_std, size=maximum_budget)
        random_order = random_rng.permutation(config.candidate_count).astype(np.int64)

        for budget in config.budgets:
            action_sets = {
                "fixed": fixed_action_indices(fixed_order, budget),
                "random": random_order[:budget],
            }
            for method, action_indices in action_sets.items():
                sensing_matrix = sensing_matrix_from_indices(candidate_dictionary, action_indices)
                _, observations = simulate_measurements(sensing_matrix, x, paired_noise[:budget])
                start = perf_counter()
                reconstruction = lasso_ista(sensing_matrix, observations, config.lasso)
                elapsed_ms = (perf_counter() - start) * 1_000.0
                records.append(
                    TrialRecord(
                        method=method,
                        budget=budget,
                        trial=trial_index,
                        nmse=normalized_mean_squared_error(x, reconstruction.coefficients),
                        support_f1=support_f1_score(true_support, reconstruction.coefficients),
                        reconstruction_ms=elapsed_ms,
                        converged=reconstruction.converged,
                    )
                )

    return BaselineStudyResult(
        config=config,
        records=tuple(records),
        summaries=_summarize(records, ("fixed", "random")),
    )


@dataclass(frozen=True)
class AdaptiveStudyConfig:
    """Frozen, paired protocol for the Stage 3 adaptive comparison."""

    signal: SignalConfig = SignalConfig()
    candidate_count: int = 128
    budgets: tuple[int, ...] = (8, 12, 16, 20, 24, 32)
    trials_per_budget: int = 100
    noise_std: float = 0.10
    lasso: LassoConfig = LassoConfig()
    policy: InformationGuidedPolicyConfig = InformationGuidedPolicyConfig()
    seed: int = 20_260_913

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
        if not np.isclose(self.noise_std, self.policy.noise_std):
            raise ValueError("policy.noise_std must equal the experiment noise_std")


@dataclass(frozen=True)
class ActionTraceRecord:
    """One adaptive decision, retained to make the policy auditable."""

    budget: int
    trial: int
    step: int
    action_index: int
    observation: float
    information_score: float


@dataclass(frozen=True)
class AdaptiveStudyResult:
    """All Stage 3 reconstructions plus auditable adaptive action histories."""

    config: AdaptiveStudyConfig
    records: tuple[TrialRecord, ...]
    summaries: tuple[SummaryRecord, ...]
    action_traces: tuple[ActionTraceRecord, ...]


def _record_reconstruction(
    records: list[TrialRecord],
    method: str,
    budget: int,
    trial_index: int,
    sensing_matrix: np.ndarray,
    observations: np.ndarray,
    x: np.ndarray,
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
            trial=trial_index,
            nmse=normalized_mean_squared_error(x, reconstruction.coefficients),
            support_f1=support_f1_score(true_support, reconstruction.coefficients),
            reconstruction_ms=elapsed_ms,
            converged=reconstruction.converged,
        )
    )


def run_adaptive_study(config: AdaptiveStudyConfig = AdaptiveStudyConfig()) -> AdaptiveStudyResult:
    """Compare adaptive selection with the same paired fixed/random protocol.

    For every trial and budget, all three methods share exactly one generated
    hidden signal and one pre-drawn noise prefix.  Only their selected rows of
    ``A`` differ.  The adaptive policy sees prior observations but never ``x``.
    """

    dictionary_seed, fixed_seed, trial_seed = np.random.SeedSequence(config.seed).spawn(3)
    dictionary = create_candidate_dictionary(
        config.candidate_count, config.signal.dimension, np.random.default_rng(dictionary_seed)
    )
    fixed_order = make_fixed_action_order(config.candidate_count, np.random.default_rng(fixed_seed))
    trial_seeds = trial_seed.spawn(config.trials_per_budget)
    maximum_budget = max(config.budgets)
    records: list[TrialRecord] = []
    traces: list[ActionTraceRecord] = []

    for trial_index, seed_sequence in enumerate(trial_seeds):
        signal_seed, noise_seed, random_seed = seed_sequence.spawn(3)
        x, true_support = generate_sparse_signal(config.signal, np.random.default_rng(signal_seed))
        paired_noise = np.random.default_rng(noise_seed).normal(
            0.0, config.noise_std, size=maximum_budget
        )
        random_order = np.random.default_rng(random_seed).permutation(config.candidate_count).astype(np.int64)

        for budget in config.budgets:
            noise_prefix = paired_noise[:budget]
            for method, action_indices in {
                "fixed": fixed_action_indices(fixed_order, budget),
                "random": random_order[:budget],
            }.items():
                sensing_matrix = sensing_matrix_from_indices(dictionary, action_indices)
                _, observations = simulate_measurements(sensing_matrix, x, noise_prefix)
                _record_reconstruction(
                    records, method, budget, trial_index, sensing_matrix, observations,
                    x, true_support, config.lasso,
                )

            sequential_run: SequentialSensingRun = run_information_guided_sensing(
                dictionary, x, noise_prefix, config.policy
            )
            _record_reconstruction(
                records, "information_guided", budget, trial_index,
                sequential_run.sensing_matrix, sequential_run.observations,
                x, true_support, config.lasso,
            )
            traces.extend(
                ActionTraceRecord(
                    budget=budget,
                    trial=trial_index,
                    step=step,
                    action_index=int(action_index),
                    observation=float(observation),
                    information_score=float(score),
                )
                for step, (action_index, observation, score) in enumerate(
                    zip(
                        sequential_run.action_indices,
                        sequential_run.observations,
                        sequential_run.selected_scores,
                        strict=True,
                    )
                )
            )

    return AdaptiveStudyResult(
        config=config,
        records=tuple(records),
        summaries=_summarize(records, ("fixed", "random", "information_guided")),
        action_traces=tuple(traces),
    )

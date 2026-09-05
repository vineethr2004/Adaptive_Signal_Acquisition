"""Stage 1 deterministic simulator for y = A x + epsilon.

This module deliberately stops before reconstruction or adaptive sensing.  It
builds the data-generating process that later stages will use and records each
array needed to explain one run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ToyConfig:
    """Configuration for one self-contained toy sensing run."""

    signal_dimension: int = 64
    sparsity: int = 4
    candidate_count: int = 128
    measurement_budget: int = 16
    noise_std: float = 0.10
    seed: int = 20_260_905

    def __post_init__(self) -> None:
        if not 0 < self.sparsity <= self.signal_dimension:
            raise ValueError("sparsity must be in [1, signal_dimension]")
        if self.candidate_count < self.measurement_budget:
            raise ValueError("candidate_count must be at least measurement_budget")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class ToyRun:
    """Every array produced in a single fixed-sensing simulation run."""

    config: ToyConfig
    x: FloatArray
    support: IntArray
    candidate_dictionary: FloatArray
    selected_action_indices: IntArray
    sensing_matrix: FloatArray
    clean_measurements: FloatArray
    noise: FloatArray
    observations: FloatArray


def _make_rngs(seed: int) -> tuple[np.random.Generator, ...]:
    """Create independent deterministic streams for each part of the simulator."""

    streams = np.random.SeedSequence(seed).spawn(4)
    return tuple(np.random.default_rng(stream) for stream in streams)


def generate_sparse_signal(
    dimension: int, sparsity: int, rng: np.random.Generator
) -> tuple[FloatArray, IntArray]:
    """Generate a vector with exactly ``sparsity`` non-zero Gaussian entries."""

    support = np.sort(rng.choice(dimension, size=sparsity, replace=False)).astype(np.int64)
    x = np.zeros(dimension, dtype=np.float64)
    x[support] = rng.normal(loc=0.0, scale=1.0, size=sparsity)
    return x, support


def create_candidate_dictionary(
    candidate_count: int, dimension: int, rng: np.random.Generator
) -> FloatArray:
    """Create normalized +/-1 measurement vectors.

    Each row is an allowed action a^T.  Row normalization gives every action
    the same sensing energy, which matters for fair later comparisons.
    """

    dictionary = rng.choice(np.array([-1.0, 1.0]), size=(candidate_count, dimension))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    return dictionary.astype(np.float64)


def select_fixed_actions(
    candidate_dictionary: FloatArray,
    measurement_budget: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    """Preselect actions without using the signal or any observed data."""

    indices = np.sort(
        rng.choice(candidate_dictionary.shape[0], size=measurement_budget, replace=False)
    ).astype(np.int64)
    return candidate_dictionary[indices], indices


def simulate_noisy_measurements(
    sensing_matrix: FloatArray, x: FloatArray, noise_std: float, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Produce clean values A @ x, additive Gaussian noise, and observations."""

    clean_measurements = sensing_matrix @ x
    noise = rng.normal(loc=0.0, scale=noise_std, size=sensing_matrix.shape[0])
    observations = clean_measurements + noise
    return clean_measurements, noise, observations


def run_toy_simulation(config: ToyConfig = ToyConfig()) -> ToyRun:
    """Run the complete Stage 1 fixed-sensing simulation deterministically."""

    signal_rng, dictionary_rng, selection_rng, noise_rng = _make_rngs(config.seed)
    x, support = generate_sparse_signal(config.signal_dimension, config.sparsity, signal_rng)
    candidate_dictionary = create_candidate_dictionary(
        config.candidate_count, config.signal_dimension, dictionary_rng
    )
    sensing_matrix, selected_action_indices = select_fixed_actions(
        candidate_dictionary, config.measurement_budget, selection_rng
    )
    clean_measurements, noise, observations = simulate_noisy_measurements(
        sensing_matrix, x, config.noise_std, noise_rng
    )
    return ToyRun(
        config=config,
        x=x,
        support=support,
        candidate_dictionary=candidate_dictionary,
        selected_action_indices=selected_action_indices,
        sensing_matrix=sensing_matrix,
        clean_measurements=clean_measurements,
        noise=noise,
        observations=observations,
    )

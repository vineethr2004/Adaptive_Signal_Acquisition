"""Measurement dictionaries and the linear noisy sensing model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def create_candidate_dictionary(
    candidate_count: int, signal_dimension: int, rng: np.random.Generator
) -> FloatArray:
    """Create a finite set of equal-energy, dense +/-1 sensing actions.

    Each row is an allowed measurement vector ``a.T``.  It has
    ``signal_dimension`` weights, one for every possible signal coordinate.
    The number of rows is the number of *available actions*, not the number
    of unknown signal values.
    """

    if candidate_count <= 0 or signal_dimension <= 0:
        raise ValueError("candidate_count and signal_dimension must be positive")
    dictionary = rng.choice(
        np.array([-1.0, 1.0]), size=(candidate_count, signal_dimension)
    ).astype(np.float64)
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    return dictionary


def validate_action_indices(
    action_indices: IntArray, candidate_count: int, measurement_budget: int
) -> None:
    """Reject invalid or repeated actions before creating a sensing matrix."""

    if action_indices.shape != (measurement_budget,):
        raise ValueError("action_indices must contain exactly measurement_budget indices")
    if len(np.unique(action_indices)) != measurement_budget:
        raise ValueError("actions must not repeat within one measurement budget")
    if np.any(action_indices < 0) or np.any(action_indices >= candidate_count):
        raise ValueError("action index is outside the candidate dictionary")


def sensing_matrix_from_indices(dictionary: FloatArray, action_indices: IntArray) -> FloatArray:
    """Stack selected candidate actions into the sensing matrix A."""

    return dictionary[action_indices].copy()


def simulate_measurements(
    sensing_matrix: FloatArray, x: FloatArray, noise: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return clean measurements ``A @ x`` and observations ``A @ x + noise``."""

    if sensing_matrix.shape[1] != x.shape[0]:
        raise ValueError("sensing_matrix and x have incompatible dimensions")
    if noise.shape != (sensing_matrix.shape[0],):
        raise ValueError("noise must have one value per measurement row")
    clean = sensing_matrix @ x
    return clean, clean + noise

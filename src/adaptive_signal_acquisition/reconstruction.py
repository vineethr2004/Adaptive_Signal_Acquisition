"""Sparse reconstruction methods.

Stage 2 uses one transparent LASSO implementation for every sensing baseline.
Keeping this reconstructor fixed lets later experiments isolate the effect of
how measurements are selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LassoConfig:
    """Settings for the LASSO objective 0.5||Az-y||² + lambda||z||₁."""

    lambda_value: float = 0.08
    max_iterations: int = 5_000
    tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.lambda_value < 0:
            raise ValueError("lambda_value must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass(frozen=True)
class LassoResult:
    """The reconstructed signal and diagnostics from the iterative solver."""

    coefficients: FloatArray
    objective_value: float
    iterations: int
    converged: bool


def soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    """Shrink weak coordinates to zero; the proximal step for the L1 penalty."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def lasso_objective(
    sensing_matrix: FloatArray, observations: FloatArray, coefficients: FloatArray, lambda_value: float
) -> float:
    """Evaluate 0.5||Az-y||² + lambda||z||₁ for one candidate signal."""

    residual = sensing_matrix @ coefficients - observations
    return float(0.5 * residual @ residual + lambda_value * np.linalg.norm(coefficients, ord=1))


def lasso_ista(
    sensing_matrix: FloatArray, observations: FloatArray, config: LassoConfig = LassoConfig()
) -> LassoResult:
    """Reconstruct a sparse signal with iterative shrinkage-thresholding.

    Each iteration first moves the candidate signal toward one that explains
    the observed measurements, then soft-thresholds weak coordinates toward
    zero.  This directly implements the two terms in the LASSO objective.
    """

    if sensing_matrix.ndim != 2:
        raise ValueError("sensing_matrix must be two-dimensional")
    measurement_count, signal_dimension = sensing_matrix.shape
    if observations.shape != (measurement_count,):
        raise ValueError("observations must have one value per measurement row")

    lipschitz_constant = float(np.linalg.norm(sensing_matrix, ord=2) ** 2)
    if lipschitz_constant == 0.0:
        raise ValueError("sensing_matrix must contain at least one non-zero row")
    step_size = 1.0 / lipschitz_constant
    coefficients = np.zeros(signal_dimension, dtype=np.float64)

    for iteration in range(1, config.max_iterations + 1):
        residual = observations - sensing_matrix @ coefficients
        gradient_step = coefficients + step_size * (sensing_matrix.T @ residual)
        updated = soft_threshold(gradient_step, step_size * config.lambda_value)
        if np.linalg.norm(updated - coefficients) <= config.tolerance * max(
            1.0, np.linalg.norm(coefficients)
        ):
            coefficients = updated
            return LassoResult(
                coefficients=coefficients,
                objective_value=lasso_objective(
                    sensing_matrix, observations, coefficients, config.lambda_value
                ),
                iterations=iteration,
                converged=True,
            )
        coefficients = updated

    return LassoResult(
        coefficients=coefficients,
        objective_value=lasso_objective(
            sensing_matrix, observations, coefficients, config.lambda_value
        ),
        iterations=config.max_iterations,
        converged=False,
    )

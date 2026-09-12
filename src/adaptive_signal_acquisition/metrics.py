"""Metrics for comparing reconstruction quality across sensing policies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def normalized_mean_squared_error(reference: FloatArray, estimate: FloatArray) -> float:
    """Return NMSE with a tiny denominator guard for an all-zero reference."""

    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.sum((reference - estimate) ** 2) / (np.sum(reference**2) + 1e-12))


def support_f1_score(
    true_support: IntArray, estimate: FloatArray, threshold: float = 1e-3
) -> float:
    """Return the F1 score for identifying active sparse-signal coordinates."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    estimated_support = set(np.flatnonzero(np.abs(estimate) > threshold).tolist())
    true_indices = set(true_support.tolist())
    if not estimated_support and not true_indices:
        return 1.0
    if not estimated_support or not true_indices:
        return 0.0
    true_positives = len(estimated_support & true_indices)
    precision = true_positives / len(estimated_support)
    recall = true_positives / len(true_indices)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))

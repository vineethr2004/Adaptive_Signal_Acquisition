"""Non-adaptive sensing strategies used as Stage 2 reference points."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]


def make_fixed_action_order(candidate_count: int, rng: np.random.Generator) -> IntArray:
    """Create one preselected action order, independent of every trial signal."""

    return rng.permutation(candidate_count).astype(np.int64)


def fixed_action_indices(fixed_order: IntArray, measurement_budget: int) -> IntArray:
    """Use the same preselected first B actions for every trial."""

    if not 0 < measurement_budget <= fixed_order.size:
        raise ValueError("measurement_budget must fit in fixed_order")
    return fixed_order[:measurement_budget].copy()


def random_action_indices(
    candidate_count: int, measurement_budget: int, rng: np.random.Generator
) -> IntArray:
    """Select B actions at random before observations are made."""

    if not 0 < measurement_budget <= candidate_count:
        raise ValueError("measurement_budget must be in [1, candidate_count]")
    return rng.permutation(candidate_count)[:measurement_budget].astype(np.int64)

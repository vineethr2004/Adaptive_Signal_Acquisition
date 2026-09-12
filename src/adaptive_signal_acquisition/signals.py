"""Sparse signal models used by every experiment stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SignalConfig:
    """Definition of a directly sparse synthetic signal."""

    dimension: int = 64
    sparsity: int = 4
    amplitude_std: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.sparsity <= self.dimension:
            raise ValueError("sparsity must be in [1, dimension]")
        if self.amplitude_std <= 0:
            raise ValueError("amplitude_std must be positive")


def generate_sparse_signal(
    config: SignalConfig, rng: np.random.Generator
) -> tuple[FloatArray, IntArray]:
    """Generate a signal with exactly ``config.sparsity`` active entries."""

    support = np.sort(
        rng.choice(config.dimension, size=config.sparsity, replace=False)
    ).astype(np.int64)
    x = np.zeros(config.dimension, dtype=np.float64)
    x[support] = rng.normal(loc=0.0, scale=config.amplitude_std, size=config.sparsity)
    return x, support

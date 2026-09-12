import numpy as np

from adaptive_signal_acquisition.measurements import (
    create_candidate_dictionary,
    sensing_matrix_from_indices,
    simulate_measurements,
)
from adaptive_signal_acquisition.signals import SignalConfig, generate_sparse_signal


def test_sparse_signal_and_measurement_equation_are_reproducible() -> None:
    signal_config = SignalConfig(dimension=20, sparsity=3)
    signal_rng = np.random.default_rng(123)
    dictionary_rng = np.random.default_rng(456)
    x, support = generate_sparse_signal(signal_config, signal_rng)
    dictionary = create_candidate_dictionary(40, signal_config.dimension, dictionary_rng)
    action_indices = np.array([0, 3, 8, 12, 21, 33, 39], dtype=np.int64)
    sensing_matrix = sensing_matrix_from_indices(dictionary, action_indices)
    noise = np.linspace(-0.1, 0.1, num=7)
    clean, observations = simulate_measurements(sensing_matrix, x, noise)

    assert x.shape == (20,)
    assert support.shape == (3,)
    assert np.count_nonzero(x) == 3
    assert sensing_matrix.shape == (7, 20)
    assert np.allclose(np.linalg.norm(dictionary, axis=1), 1.0)
    assert np.allclose(observations, sensing_matrix @ x + noise)
    assert np.allclose(clean + noise, observations)

import numpy as np

from adaptive_signal_acquisition import ToyConfig, run_toy_simulation


def test_same_seed_reproduces_every_array() -> None:
    config = ToyConfig(seed=1234)
    first = run_toy_simulation(config)
    second = run_toy_simulation(config)

    for name in (
        "x",
        "support",
        "candidate_dictionary",
        "selected_action_indices",
        "sensing_matrix",
        "clean_measurements",
        "noise",
        "observations",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_measurement_equation_and_shapes() -> None:
    config = ToyConfig(signal_dimension=20, sparsity=3, candidate_count=40, measurement_budget=7)
    run = run_toy_simulation(config)

    assert run.x.shape == (20,)
    assert run.sensing_matrix.shape == (7, 20)
    assert run.observations.shape == (7,)
    assert np.count_nonzero(run.x) == 3
    assert np.allclose(run.observations, run.sensing_matrix @ run.x + run.noise)
    assert np.allclose(np.linalg.norm(run.candidate_dictionary, axis=1), 1.0)


def test_different_seed_changes_the_run() -> None:
    first = run_toy_simulation(ToyConfig(seed=1))
    second = run_toy_simulation(ToyConfig(seed=2))
    assert not np.array_equal(first.x, second.x)

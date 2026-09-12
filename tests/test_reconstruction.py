import numpy as np

from adaptive_signal_acquisition.reconstruction import LassoConfig, lasso_ista


def test_lasso_matches_soft_threshold_solution_for_identity_matrix() -> None:
    sensing_matrix = np.eye(3)
    observations = np.array([1.0, 0.04, -2.0])
    result = lasso_ista(
        sensing_matrix,
        observations,
        LassoConfig(lambda_value=0.1, tolerance=1e-10),
    )

    assert result.converged
    assert np.allclose(result.coefficients, np.array([0.9, 0.0, -1.9]), atol=1e-8)


def test_lasso_returns_correct_shape_for_underdetermined_measurements() -> None:
    sensing_matrix = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    observations = np.array([1.0, 1.0])
    result = lasso_ista(sensing_matrix, observations, LassoConfig(lambda_value=0.05))

    assert result.coefficients.shape == (3,)
    assert result.objective_value >= 0.0

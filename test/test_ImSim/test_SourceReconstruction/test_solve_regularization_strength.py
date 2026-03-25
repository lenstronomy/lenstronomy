import numpy as np
import pytest
from lenstronomy.ImSim.SourceReconstruction.solve_regularization_strength import (
    d_log_evi_d_lambda,
    solve_optimal_lambda,
)


def d_evidence_standard(l, num_reg_shape, Cv, M, b):
    Lambda = l * Cv
    M_Lambda = M + Lambda
    MLambda_inv = np.linalg.inv(M_Lambda)
    Minv_b = np.matmul(MLambda_inv, b)
    CvMinv_b = np.matmul(Cv, Minv_b)
    bmb = np.sum(Minv_b * CvMinv_b)
    Minv_Cv = np.matmul(MLambda_inv, Cv)
    trace_minv = np.trace(Minv_Cv)
    d_evi = -bmb + num_reg_shape / l - trace_minv
    return d_evi


# test the ouput of d_log_evi_d_lambda
def test_d_log_evi_d_lambda():
    # test on 10x10 matrices
    # Zeroth-order regularization
    M = np.diag(np.arange(1, 10 + 1) + 0.1) + 0.1 * np.random.rand(10, 10)
    M = (M + M.T) / 2  # Ensure M is symmetric
    U = np.identity(10)  # Zeroth-order regularization
    b = np.random.rand(10)
    l = 1.0
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)

    l = 1e2
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)

    l = 1e-2
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)

    # Gradient regularization
    M = np.diag(np.arange(1, 10 + 1) + 0.1) + 0.1 * np.random.rand(10, 10)
    M = (M + M.T) / 2  # Ensure M is symmetric
    U = np.array(
        [
            [4.0, -1.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -0.0, 4.0, -1.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0, -1.0, -1.0, 4.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -0.0, 4.0, -1.0, -1.0, -0.0, 0.0, 0.0],
            [0.0, 0.0, -0.0, -1.0, -1.0, 4.0, -0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, -0.0, 4.0, -1.0, -1.0, -0.0],
            [0.0, 0.0, 0.0, 0.0, -0.0, -1.0, -1.0, 4.0, -0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -1.0, -1.0, 4.0],
        ]
    )
    b = np.random.rand(10)
    l = 1.0
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)

    l = 1e2
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)

    l = 1e-2
    result = d_log_evi_d_lambda(l, U, M, b)
    expected_d_evi = d_evidence_standard(l, 10, U, M, b)
    assert np.isclose(result, expected_d_evi, rtol=1e-6)


# test solve_optimal_lambda
def test_solve_optimal_lambda_success():
    M = np.array(
        [
            [
                8.14707500e07,
                4.92308930e07,
                2.06435317e07,
                4.92308930e07,
                9.45135773e07,
                4.06939303e07,
                2.06435317e07,
                4.06939303e07,
                3.62344295e07,
            ],
            [
                4.92308930e07,
                5.69418884e07,
                4.30476303e07,
                4.42477596e07,
                1.00579268e08,
                5.02013394e07,
                3.56392359e07,
                5.13022983e07,
                4.10497154e07,
            ],
            [
                2.06435317e07,
                4.30476303e07,
                6.76644230e07,
                3.56392359e07,
                9.22054832e07,
                6.37186254e07,
                3.86201273e07,
                4.92086043e07,
                3.13260726e07,
            ],
            [
                4.92308930e07,
                4.42477596e07,
                3.56392359e07,
                5.69418884e07,
                1.00579268e08,
                5.13022983e07,
                4.30476303e07,
                5.02013394e07,
                4.10497154e07,
            ],
            [
                9.45135773e07,
                1.00579268e08,
                9.22054832e07,
                1.00579268e08,
                2.20597354e08,
                1.22349790e08,
                9.22054832e07,
                1.22349790e08,
                1.01946283e08,
            ],
            [
                4.06939303e07,
                5.02013394e07,
                6.37186254e07,
                5.13022983e07,
                1.22349790e08,
                8.70143665e07,
                4.92086043e07,
                6.91760519e07,
                6.72341702e07,
            ],
            [
                2.06435317e07,
                3.56392359e07,
                3.86201273e07,
                4.30476303e07,
                9.22054832e07,
                4.92086043e07,
                6.76644230e07,
                6.37186254e07,
                3.13260726e07,
            ],
            [
                4.06939303e07,
                5.13022983e07,
                4.92086043e07,
                5.02013394e07,
                1.22349790e08,
                6.91760519e07,
                6.37186254e07,
                8.70143665e07,
                6.72341702e07,
            ],
            [
                3.62344295e07,
                4.10497154e07,
                3.13260726e07,
                4.10497154e07,
                1.01946283e08,
                6.72341702e07,
                3.13260726e07,
                6.72341702e07,
                1.06874468e08,
            ],
        ]
    )

    U = np.array(
        [
            [4.0, -1.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0],
            [-1.0, 4.0, -1.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 4.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0],
            [-1.0, -0.0, -0.0, 4.0, -1.0, 0.0, -1.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0, -1.0, 4.0, -1.0, -0.0, -1.0, -0.0],
            [-0.0, -0.0, -1.0, 0.0, -1.0, 4.0, -0.0, -0.0, -1.0],
            [0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 4.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -0.0, -1.0, -0.0, -1.0, 4.0, -1.0],
            [0.0, 0.0, 0.0, -0.0, -0.0, -1.0, 0.0, -1.0, 4.0],
        ]
    )

    b = np.array(
        [
            654346.55011501,
            721038.11578404,
            685780.1280977,
            715848.44728503,
            1571888.15961561,
            844689.32636277,
            679398.88829259,
            865511.88326071,
            653260.41790548,
        ]
    )

    lower_bound, upper_bound = 1e-1, 1e6

    optimal_lambda = solve_optimal_lambda(
        d_log_evi_d_lambda,
        U,
        M,
        b,
        lower_bound,
        upper_bound,
        tolerance=1e2,
        max_iterations=20,
    )
    assert np.isclose(optimal_lambda, 14801.12391052246, atol=1e-5)

    optimal_lambda = solve_optimal_lambda(
        d_log_evi_d_lambda,
        U,
        M,
        b,
        lower_bound,
        upper_bound,
        tolerance=1e-5,
        max_iterations=20,
    )
    assert np.isclose(optimal_lambda, 14813.044838285445, atol=1e-5)

    optimal_lambda = solve_optimal_lambda(
        d_log_evi_d_lambda,
        U,
        M,
        b,
        lower_bound,
        upper_bound,
        tolerance=1e-5,
        max_iterations=10,
    )
    assert np.isclose(optimal_lambda, 15136.817236328123, atol=1e-5)

    # Tests for check_initial_bounds parameter
    with pytest.raises(
        ValueError,
        match="`initial_lower_bound` must be strictly less than `initial_upper_bound`.",
    ):
        solve_optimal_lambda(
            d_log_evi_d_lambda, U, M, b, 6.0, 5.0, check_initial_bounds=True
        )

    with pytest.raises(
        ValueError, match="Derivative at `initial_lower_bound`.*expected > 0"
    ):
        solve_optimal_lambda(
            d_log_evi_d_lambda, U, M, b, 5e5, 1e6, check_initial_bounds=True
        )

    with pytest.raises(
        ValueError, match="Derivative at `initial_upper_bound`.*expected < 0"
    ):
        solve_optimal_lambda(
            d_log_evi_d_lambda, U, M, b, 1e0, 1e2, check_initial_bounds=True
        )

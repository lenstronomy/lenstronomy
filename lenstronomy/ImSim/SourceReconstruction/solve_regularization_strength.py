import numpy as np
from typing import Callable


def d_log_evi_d_lambda(l: float, U: np.ndarray, M: np.ndarray, b: np.ndarray) -> float:
    """Computes the derivative of the logarithm of the Bayesian evidence with respect to
    the regularization strength (lambda, l).

    This function calculates the derivative as:
    d(ln(Evidence))/d(lambda) ~ N_s/lambda - tr[(M+lambda*U)^-1 * U] - b^T * (M+lambda*U)^-1 * U * (M+lambda*U)^-1 * b

    Where:
    - N_s: Number of source pixels (U.shape[0])
    - lambda: The regularization strength
    - U: The regularization matrix
    - M: The M matrix
    - b: The b vector

    :param l: The current value of the regularization strength (lambda).
    :param U: The regularization matrix (numpy.ndarray).
    :param M: The M matrix (numpy.ndarray).
    :param b: The b vector (numpy.ndarray).
    :return: The computed derivative value (float).
    """
    N_source = U.shape[0]
    lambda_U = l * U
    M_plus_lambda_U = M + lambda_U

    # Compute the inverse of (M + lambda * U)
    M_plus_lambda_U_inv = np.linalg.inv(M_plus_lambda_U)

    # Compute the trace term: tr[(M+lambda*U)^-1 * U]
    trace_term_matrix = np.matmul(M_plus_lambda_U_inv, U)
    trace_term = np.trace(trace_term_matrix)

    # Compute the quadratic term: b^T * (M+lambda*U)^-1 * U * (M+lambda*U)^-1 * b
    M_plus_lambda_U_inv_b = np.matmul(M_plus_lambda_U_inv, b)
    U_times_M_plus_lambda_U_inv_b = np.matmul(U, M_plus_lambda_U_inv_b)

    # Using np.sum(v1 * v2) is equivalent to v1^T @ v2 for 1D vectors
    quadratic_term = np.sum(M_plus_lambda_U_inv_b * U_times_M_plus_lambda_U_inv_b)

    derivative_value = (N_source / l) - trace_term - quadratic_term
    return derivative_value


def solve_optimal_lambda(
    derivative_function: Callable[[float, np.ndarray, np.ndarray, np.ndarray], float],
    U: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    initial_lower_bound: float,
    initial_upper_bound: float,
    tolerance: float = 1e-7,
    max_iterations: int = 20,
    check_initial_bounds: bool = True,
) -> float:
    """Finds the optimal regularization strength (lambda) by solving for the root of the
    log-evidence derivative using a bisection method.

    The optimal lambda is typically the value where the derivative of the
    log-evidence is zero. This function assumes that the derivative
    `d(ln(Evidence))/d(lambda)` is monotonically decreasing and crosses zero
    within the specified bounds.

    :param derivative_function: A callable function that computes the derivative
                                d(ln(Evidence))/d(lambda). It must accept
                                (regularization_strength, data_matrix, regularization_matrix, data_vector)
                                as its arguments.
    :param U: The regularization matrix (numpy.ndarray).
    :param M: The M matrix (numpy.ndarray).
    :param b: The b vector (numpy.ndarray).
    :param initial_lower_bound: The lower bound for the search range of lambda.
                                It is expected that `derivative_function(initial_lower_bound, ...)` > 0.
    :param initial_upper_bound: The upper bound for the search range of lambda.
                                It is expected that `derivative_function(initial_upper_bound, ...)` < 0.
    :param tolerance: float, The desired absolute tolerance for the lambda value.
                      The search stops when the width of the search interval is less than this value.
                      Defaults to 1e-7.
    :param max_iterations: int, The maximum number of bisection iterations to perform.
                           Defaults to 20.
    :param check_initial_bounds: bool, If True, perform checks to ensure that
                                 `initial_lower_bound` < `initial_upper_bound` and
                                 that the derivative function returns the expected
                                 signs at the boundaries (positive at lower, negative at upper).
                                 Setting this to False can speed up repeated calls
                                 if the bounds are guaranteed to be valid, but
                                 disables critical error checking. Defaults to True.
    :return: float, The optimized regularization strength (lambda) that maximizes the log-evidence.
    :raises ValueError: If `check_initial_bounds` is True and `initial_lower_bound`
                        is not strictly less than `initial_upper_bound`,
                        or if the derivative function does not yield the expected signs
                        at the initial bounds (i.e., the root is not bracketed).
    """
    if check_initial_bounds:
        if not (initial_lower_bound < initial_upper_bound):
            raise ValueError(
                "`initial_lower_bound` must be strictly less than `initial_upper_bound`."
            )

        # Check initial conditions to ensure the root is bracketed
        # For a monotonically decreasing derivative crossing zero:
        # derivative at lower bound should be positive
        # derivative at upper bound should be negative
        derivative_at_lower_bound = derivative_function(initial_lower_bound, U, M, b)
        derivative_at_upper_bound = derivative_function(initial_upper_bound, U, M, b)

        if derivative_at_lower_bound <= 0:
            raise ValueError(
                f"Derivative at `initial_lower_bound` ({initial_lower_bound}) is {derivative_at_lower_bound} "
                f"(expected > 0). The root might not be bracketed correctly or is outside this range."
            )
        if derivative_at_upper_bound >= 0:
            raise ValueError(
                f"Derivative at `initial_upper_bound` ({initial_upper_bound}) is {derivative_at_upper_bound} "
                f"(expected < 0). The root might not be bracketed correctly or is outside this range."
            )

    current_lower_bound = initial_lower_bound
    current_upper_bound = initial_upper_bound

    for iteration_count in range(max_iterations):
        # Check for convergence based on interval width
        if np.abs(current_upper_bound - current_lower_bound) < tolerance:
            break

        mid_point_lambda = (current_upper_bound + current_lower_bound) / 2
        derivative_at_mid_point = derivative_function(mid_point_lambda, U, M, b)

        if derivative_at_mid_point < 0:
            # The root is in the lower half of the current interval
            current_upper_bound = mid_point_lambda
        elif derivative_at_mid_point > 0:
            # The root is in the upper half of the current interval
            current_lower_bound = mid_point_lambda

    # Return the midpoint of the final interval as the approximate optimal lambda
    return (current_lower_bound + current_upper_bound) / 2

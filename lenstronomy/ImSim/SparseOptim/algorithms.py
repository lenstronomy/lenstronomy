__author__ = 'aymgal'

# implementations of gradient descent and proximal algorithms

import numpy as np


def step_FB(x, grad, prox, step_size):
    """Forward Backward Alorithm"""
    x_next = prox(x - step_size * grad(x), step_size)
    return x_next


def step_FISTA(x, y, t, grad, prox, step_size):
    """Fast Iterative Shrinkage-Thresholding Alorithm"""
    x_next = prox(y - step_size * grad(y), step_size)
    t_next = 0.5 * (1 + np.sqrt(1 + 4*t**2))
    factor = (t - 1) / t_next
    y_next = x_next + factor * (x_next - x)
    return x_next, y_next, t_next

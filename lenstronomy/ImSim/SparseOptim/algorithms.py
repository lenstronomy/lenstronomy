# implementations of gradient descent and proximal algorithms


def FB_step(x, prox, grad, step_size):
    """Forward Backward Alorithm"""
    x_next = x - step_size * grad(x)
    x_next = prox(x_next)
    return x_next


def FISTA_step(x, y, t, prox, grad, step_size):
    """Fast Iterative Shrinkage-Thresholding Alorithm"""
    x_next = prox(y - step_size * grad(y), step_size)
    t_next = 0.5 * (1 + np.sqrt(1 + 4*t**2))
    factor = (t - 1) / t_next
    y_next = x_next + factor * (x_next - x)
    return x_next, y_next, t_next


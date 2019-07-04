__author__ = 'aymgal'

import numpy as np
from scipy import special


# transform the unit hypercube to pysical parameters for (nested) sampling


SQRT2 = np.sqrt(2)


def unit2gaussian(x, mu, sigma):
    """
    mapping from uniform distribution on unit hypercube
    to truncated gaussian distribution on parameter space, 
    with mean 'mu' and std dev 'sigma'

    from Handley+15, eq. (A9)
    """
    return mu + SQRT2 * sigma * special.erfinv(2*x - 1)


def unit2uniform(x, vmin, vmax):
    """
    mapping from uniform distribution on parameter space 
    to uniform distribution on unit hypercube
    """
    return vmin + (vmax - vmin) * x


def uniform2unit(theta, vmin, vmax):
    """
    mapping from uniform distribution on unit hypercube
    to uniform distribution on parameter space
    """
    return (theta - vmin) / (vmax - vmin)


def cube2args_uniform(cube, lowers, uppers, num_dims, copy=False):
    """
    mapping from uniform distribution on unit hypercube 'cube'
    to uniform distribution on parameter space

    :param cube: list or 1D-array of parameter values on unit hypercube
    :param lowers: lower bounds for each parameter
    :param uppers: upper bounds for each parameter
    :param num_dims: parameter space dimension (= number of parameters)
    :param copy: If False, this function modifies 'cube' in-place. Default to False.
    :return: hypercube mapped to parameters space
    """
    if copy:
        cube_ = cube
        cube = np.zeros_like(cube_)
    for i in range(num_dims):
        val = cube_[i] if copy else cube[i]
        low, upp = lowers[i], uppers[i]
        cube[i] = unit2uniform(val, low, upp)
    return cube


def cube2args_gaussian(cube, lowers, uppers, means, sigmas, num_dims, copy=False):
    """
    mapping from uniform distribution on unit hypercube 'cube'
    to truncated gaussian distribution on parameter space, 
    with mean 'mu' and std dev 'sigma'

    :param cube: list or 1D-array of parameter values on unit hypercube
    :param lowers: lower bounds for each parameter
    :param uppers: upper bounds for each parameter
    :param means: gaussian mean for each parameter
    :param sigmas: gaussian std deviation for each parameter
    :param num_dims: parameter space dimension (= number of parameters)
    :param copy: If False, this function modifies 'cube' in-place. Default to False.
    :return: hypercube mapped to parameters space
    """
    if copy:
        cube_ = cube
        cube = np.zeros_like(cube_)
    for i in range(num_dims):
        val = cube_[i] if copy else cube[i]
        val = unit2gaussian(val, means[i], sigmas[i])
        low, upp = lowers[i], uppers[i]
        if val <= low: cube[i] = low
        elif val >= upp: cube[i] = upp
        else: cube[i] = val
    return cube


def scale_limits(lowers, uppers, scale):
    if not isinstance(lowers, np.ndarray):
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)
    mid_points = (lowers + uppers) / 2.
    widths_scaled = (uppers - lowers) * scale
    lowers_scaled = mid_points - widths_scaled / 2.
    uppers_scaled = mid_points + widths_scaled / 2.
    return lowers_scaled, uppers_scaled

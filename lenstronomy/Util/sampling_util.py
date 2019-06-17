__author__ = 'aymgal'

import numpy as np
from scipy import special


# transform the unit hypercube to pysical parameters for sampling


SQRT2 = np.sqrt(2)


def unit2gaussian(x, mu, sigma):
    """from Handley+15 (PolyChord paper)"""
    return mu + SQRT2 * sigma * special.erfinv(2*x - 1)


def unit2uniform(x, vmin, vmax):
    """from Handley+15 (PolyChord paper)"""
    return vmin + (vmax - vmin) * x


def uniform2unit(theta, vmin, vmax):
    """from Handley+15 (PolyChord paper)"""
    return (theta - vmin) / (vmax - vmin)


def cube2args_uniform(cube, lowers, uppers, num_dims, copy=False):
    """copy = False leads to altering the cube 'in-place'"""
    if copy:
        cube_ = cube
        cube = np.zeros_like(cube_)
    for i in range(num_dims):
        val = cube_[i] if copy else cube[i]
        low, upp = lowers[i], uppers[i]
        cube[i] = unit2uniform(val, low, upp)
    return cube


def cube2args_gaussian(cube, lowers, uppers, means, sigmas, num_dims, copy=False):
    """copy = False leads to altering the cube 'in-place'"""
    if copy:
        cube_ = cube
        cube = np.zeros_like(cube_)
    for i in range(num_dims):
        val = cube_[i] if copy else cube[i]
        low, upp = lowers[i], uppers[i]
        if val <= low: cube[i] = low
        elif val >= upp: cube[i] = upp
        else: cube[i] = unit2gaussian(val, means[i], sigmas[i])
    return cube

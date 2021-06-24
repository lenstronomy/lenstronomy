__author__ = 'aymgal'

import numpy as np
from scipy import stats

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


# transform the unit hypercube to pysical parameters for (nested) sampling

@export
def unit2uniform(x, vmin, vmax):
    """
    mapping from uniform distribution on parameter space 
    to uniform distribution on unit hypercube
    """
    return vmin + (vmax - vmin) * x


@export
def uniform2unit(theta, vmin, vmax):
    """
    mapping from uniform distribution on unit hypercube
    to uniform distribution on parameter space
    """
    return (theta - vmin) / (vmax - vmin)


@export
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

@export
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
    a, b = (np.array(lowers)-means)/sigmas, (np.array(uppers)-means)/sigmas
    cube[:] = stats.truncnorm.ppf(cube_ if copy else cube, a=a, b=b, loc=means, scale=sigmas)
    return cube


@export
def scale_limits(lowers, uppers, scale):
    if not isinstance(lowers, np.ndarray):
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)
    mid_points = (lowers + uppers) / 2.
    widths_scaled = (uppers - lowers) * scale
    lowers_scaled = mid_points - widths_scaled / 2.
    uppers_scaled = mid_points + widths_scaled / 2.
    return lowers_scaled, uppers_scaled


@export
def sample_ball(p0, std, size=1, dist='uniform'):
    """
    Produce a ball of walkers around an initial parameter value.
    this routine is from the emcee package as it became deprecated there

    :param p0: The initial parameter values (array).
    :param std: The axis-aligned standard deviation (array).
    :param size: The number of samples to produce.
    :param dist: string, specifies the distribution being sampled, supports 'uniform' and 'normal'

    """
    assert(len(p0) == len(std))
    if dist == 'uniform':
        return np.vstack([p0 + std * np.random.uniform(low=-1, high=1, size=len(p0))
                         for i in range(size)])
    elif dist == 'normal':
        return np.vstack([p0 + std * np.random.normal(loc=0, scale=1, size=len(p0))
                          for i in range(size)])
    else:
        raise ValueError('distribution %s not supported. Chose among "uniform" or "normal".' % dist)

__author__ = "ajshajib", "sibirrer"
"""
Multi-Gaussian expansion fitting, based on Capellari 2002, http://adsabs.harvard.edu/abs/2002MNRAS.333..400C
"""

import numpy as np
from scipy.optimize import nnls
import warnings
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
gaussian_func = Gaussian()


def gaussian(R, sigma, amp):
    """

    :param R: radius
    :param sigma: gaussian sigma
    :param amp: normalization
    :return: Gaussian function
    """
    c = amp / (2 * np.pi * sigma**2)
    return c * np.exp(-(R/float(sigma))**2/2.)


def mge_1d(r_array, flux_r, N=20, linspace=False):
    """

    :param r_array: list or radii (numpy array)
    :param flux_r: list of flux values (numpy array)
    :param N: number of Gaussians
    :return: amplitudes and Gaussian sigmas for the best 1d flux profile
    """
    if N == 0:
        warnings.warn('Number of MGE went down to zero! This should not happen!', Warning)
        amplitudes = [0]
        sigmas = [1]
        norm = 0
        return amplitudes, sigmas, norm
    try:
        amplitudes, sigmas, norm = _mge_1d(r_array, flux_r, N, linspace=linspace)
    except:
        N_new = N - 1
        amplitudes, sigmas, norm = mge_1d(r_array, flux_r, N=N_new, linspace=linspace)
    return amplitudes, sigmas, norm


def _mge_1d(r_array, flux_r, N=20, linspace=False):
    """

    :param r_array:
    :param flux_r:
    :param N:
    :return:
    """
    if linspace is True:
        sigmas = np.linspace(r_array[0], r_array[-1] / 2, N + 2)[1:-1]
    else:
        sigmas = np.logspace(np.log10(r_array[0]), np.log10((r_array[-1] + 0.0000001) / 2.), N + 2)[1:-1]
    # sigmas = np.linspace(r_array[0], r_array[-1]/2, N + 2)[1:-1]

    A = np.zeros((len(flux_r), N))
    for j in np.arange(A.shape[1]):
        A[:, j] = gaussian(r_array, sigmas[j], 1.)
    amplitudes, norm = nnls(A, flux_r)
    return amplitudes, sigmas, norm


def de_projection_3d(amplitudes, sigmas):
    """
    de-projects a gaussian (or list of multiple Gaussians from a 2d projected to a 3d profile)
    :param amplitudes:
    :param sigmas:
    :return:
    """
    amplitudes_3d = amplitudes / sigmas / np.sqrt(2*np.pi)
    return amplitudes_3d, sigmas
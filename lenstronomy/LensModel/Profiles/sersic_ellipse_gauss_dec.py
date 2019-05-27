# -*- coding: utf-8 -*-
"""
This module contains `class SersicEllipseGaussDec()` to compute the
lensing properties of a Sersic profile using the Gaussian decomposition method
of Shajib (2019).
"""

__author__ = 'ajshajib'

import numpy as np
from scipy.special import comb

from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.LensModel.Profiles.gauss_decomposition import GaussDecomposition


class SersicEllipseGaussDec(object):
    """
    class for Sersic profile convergence using Gauss expansion
    """
    param_names = ['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x',
                   'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5,
                           'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def __init__(self, n_sigma=15, sigma_start_mult=0.02, sigma_end_mult=15,
                 precision=10):
        """
        Set up settings for the Gaussian decomposition. For more details about
        the decomposition parameters, see Shajib (2019).
        :param n_sigma: Number of Gaussian components.
        :type n_sigma:
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas.
        :type sigma_start_mult:
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas.
        :type sigma_end_mult:
        :param precision: Numerical precision of Gaussian decomposition.
        :type precision:
        """
        self.gauss_decomposition = GaussDecomposition()
        self.util = SersicUtil()

        self.n_sigma = n_sigma
        self.sigma_start_mult = sigma_start_mult
        self.sigma_end_mult = sigma_end_mult
        self.precision = precision

        p = self.precision
        # nodes and weights based on Fourier-Euler method
        # for details Abate & Whitt (2006)
        kes = np.arange(2 * p + 1)
        self.betas = np.sqrt(2 * p * np.log(10) / 3. + 2 * 1j * np.pi * kes)
        epsilons = np.zeros(2 * p + 1)

        epsilons[0] = 0.5
        epsilons[1:p + 1] = 1.
        epsilons[-1] = 1 / 2 ** p

        for k in range(1, p):
            epsilons[2 * p - k] = epsilons[2 * p - k + 1] + 1 / 2 ** p * comb(
                p, k)

        self.etas = (-1) ** kes * epsilons * 10 ** (p / 3) * 2 * np.sqrt(2 *
                                                                      np.pi)

    def get_amps(self, n_sersic, R_sersic, k_eff):
        """
        Compute the amplitudes and sigmas of Gaussian components using the
        integral transform with Gaussian kernel from Shajib (2019). The
        returned values are in the convention of eq. (2.13).
        :param x:
        :type x:
        :param y:
        :type y:
        :param theta_E:
        :type theta_E:
        :param gamma:
        :type gamma:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        sigma_start = self.sigma_start_mult*R_sersic
        sigma_end = self.sigma_end_mult*R_sersic

        sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_end),
                           self.n_sigma)

        f_sigmas = np.sum(self.etas * self.kappa_y(
                                sigmas[:,np.newaxis]*self.betas[np.newaxis, :],
                                n_sersic, R_sersic, k_eff).real,
                          axis=1
                          )

        # weighting for trapezoid method integral
        f_sigmas[0] *= 0.5
        f_sigmas[-1] *= 0.5

        del_log_sigma = np.abs(np.diff(np.log(sigmas)).mean())

        f_sigmas *= del_log_sigma / np.sqrt(2*np.pi)

        return f_sigmas, sigmas

    def kappa_y(self, y, n_sersic, R_sersic, k_eff):
        """
        Compute the profile along the minor axis.
        :param y:
        :type y:
        :param k_eff:
        :type k_eff:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        bn = self.util.b_n(n_sersic)

        return k_eff * np.exp(-bn * (y / R_sersic) ** (1. / n_sersic) + bn)

    def function(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0,
                 center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param k_eff:
        :type k_eff:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        amps, sigmas = self.get_amps(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2*np.pi * sigmas * sigmas

        return self.gauss_decomposition.function(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def derivatives(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0,
                    center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param k_eff:
        :type k_eff:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        amps, sigmas = self.get_amps(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2 * np.pi * sigmas * sigmas

        return self.gauss_decomposition.derivatives(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0,
                center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param k_eff:
        :type k_eff:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        amps, sigmas = self.get_amps(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2 * np.pi * sigmas * sigmas

        return self.gauss_decomposition.hessian(x, y, amps, sigmas, e1, e2,
                                                center_x, center_y)

    def density_2d(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0,
                   center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param k_eff:
        :type k_eff:
        :param R_sersic:
        :type R_sersic:
        :param n_sersic:
        :type n_sersic:
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :return:
        :rtype:
        """
        amps, sigmas = self.get_amps(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2 * np.pi * sigmas * sigmas

        return self.gauss_decomposition.density_2d(x, y, amps, sigmas, e1, e2,
                                                   center_x, center_y)
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
    This class computes the lensing properties of an elliptical Sersic
    profile using the Gauss decomposition method from Shajib (2019).
    """
    param_names = ['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x',
                   'center_y']
    lower_limit_default = {'amp': 0., 'R_sersic': 0., 'n_sersic': 0.5,
                           'e1': -0.5, 'e2': -0.5, 'center_x': -100.,
                           'center_y': -100.}
    upper_limit_default = {'amp': 100., 'R_sersic': 100., 'n_sersic': 8.,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100.,
                           'center_y': 100.}

    def __init__(self, n_sigma=15, sigma_start_mult=0.02, sigma_end_mult=15.,
                 precision=10, use_scipy_wofz=True, min_ellipticity=1e-5):
        """
        Set up settings for the Gaussian decomposition. For more details about
        the decomposition parameters, see Shajib (2019).
        :param n_sigma: Number of Gaussian components.
        :type n_sigma: int
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas.
        :type sigma_start_mult:float
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas.
        :type sigma_end_mult: float
        :param precision: Numerical precision of Gaussian decomposition.
        :type precision: int
        :param use_scipy_wofz: To be passed to `class GaussianEllipseKappa(
        )`. If True, Gaussian lensing will use `scipy.special.wofz`
        function. Set False for lower precision, but faster speed.
        :type use_scipy_wofz: bool
        :param min_ellipticity: To be passed to `class GaussianEllipseKappa(
        )`. Minimum ellipticity for Gaussian elliptical lensing calculation.
        For lower ellipticity than min_ellipticity the equations for the
        spherical case will be used.
        :type min_ellipticity: float
        """
        self.gauss_decomposition = GaussDecomposition(
                                            use_scipy_wofz=use_scipy_wofz,
                                            min_ellipticity=min_ellipticity)
        self.util = SersicUtil()

        self.n_sigma = n_sigma
        self.sigma_start_mult = sigma_start_mult
        self.sigma_end_mult = sigma_end_mult
        self.precision = precision

        p = self.precision
        # nodes and weights based on Fourier-Euler method
        # for details Abate & Whitt (2006)
        kes = np.arange(2 * p + 1)
        self.betas = np.sqrt(2 * p * np.log(10) / 3. + 2. * 1j * np.pi * kes)
        epsilons = np.zeros(2 * p + 1)

        epsilons[0] = 0.5
        epsilons[1:p + 1] = 1.
        epsilons[-1] = 1 / 2. ** p

        for k in range(1, p):
            epsilons[2 * p - k] = epsilons[2 * p - k + 1] + 1 / 2. ** p * comb(
                p, k)

        self.etas = (-1.) ** kes * epsilons * 10. ** (p / 3.) * 2. * \
                    np.sqrt(2. * np.pi)

    def gauss_decompose_sersic(self, n_sersic, R_sersic, k_eff):
        """
        Compute the amplitudes and sigmas of Gaussian components using the
        integral transform with Gaussian kernel from Shajib (2019). The
        returned values are in the convention of eq. (2.13).
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :return: Amplitudes and standard deviations of the Gaussian components.
        :rtype: tuple of (numpy.array, numpy.array)
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

        f_sigmas *= del_log_sigma / np.sqrt(2.*np.pi)

        return f_sigmas, sigmas

    def kappa_y(self, y, n_sersic, R_sersic, k_eff):
        """
        Compute the spherical Sersic profile at $y$.
        :param y: y coordinate.
        :type y: float or numpy.array
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic scale radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :return: Sersic function at $y$.
        :rtype: type(y)
        """
        bn = self.util.b_n(n_sersic)

        return k_eff * np.exp(-bn * (y / R_sersic) ** (1. / n_sersic) + bn)

    def function(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0.,
                 center_y=0.):
        """
        Compute the deflection potential of a Gauss-decomposed
        elliptic Sersic convergence.
        :param x: x coordinate.
        :type x: float
        :param y: y coordinate.
        :type y: float
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic scale radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :param e1: Ellipticity parameter 1.
        :type e1: float
        :param e2: Ellipticity parameter 2.
        :type e2: float
        :param center_x: x coordinate of centroid.
        :type center_x: float
        :param center_y: y coordinate of centroid.
        :type center_y: float
        :return: Deflection potential.
        :rtype: float
        """
        amps, sigmas = self.gauss_decompose_sersic(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.*np.pi * sigmas * sigmas

        return self.gauss_decomposition.function(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def derivatives(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0.,
                    center_y=0.):
        """
        Compute the derivatives of the deflection potential df/dx, df/dy for a
        Gauss-decomposed elliptic Sersic convergence.
        :param x: x coordinate.
        :type x: float or numpy.array
        :param y: y coordinate.
        :type y: float or numpy.array
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic scale radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :param e1: Ellipticity parameter 1.
        :type e1: float
        :param e2: Ellipticity parameter 2.
        :type e2: float
        :param center_x: x coordinate of centroid.
        :type center_x: float
        :param center_y: y coordinate of centroid.
        :type center_y: float
        :return: Derivatives of deflection potential.
        :rtype: tuple (type(x), type(x))
        """
        amps, sigmas = self.gauss_decompose_sersic(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gauss_decomposition.derivatives(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0.,
                center_y=0.):
        """
        Compute the Hessian of the deflection potential d^2f/dx^2,
        d^2f/dy^2, d^f/dxdy of a Gauss-decomposed elliptic Sersic convergence.
        :param x: x coordinate.
        :type x: float or numpy.array
        :param y: y coordinate.
        :type y: float or numpy.arry
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic scale radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :param e1: Ellipticity parameter 1.
        :type e1: float
        :param e2: Ellipticity parameter 2.
        :type e2: float
        :param center_x: x coordinate of centroid.
        :type center_x: float
        :param center_y: y coordinate of centroid.
        :type center_y: float
        :return: Hessian of deflection potential.
        :rtype: tuple (type(x), type(x), type(x))
        """
        amps, sigmas = self.gauss_decompose_sersic(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gauss_decomposition.hessian(x, y, amps, sigmas, e1, e2,
                                                center_x, center_y)

    def density_2d(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0.,
                   center_y=0.):
        """
        Compute the convergence profile for Gauss-decomposed
        elliptic Sersic profile.
        :param x: x coordinate.
        :type x: float or numpy.array
        :param y: y coordinate.
        :type y: float or numpy.array
        :param n_sersic: Sersic index.
        :type n_sersic: float
        :param R_sersic: Sersic scale radius.
        :type R_sersic: float
        :param k_eff: Sersic convergence at R_sersic.
        :type k_eff: float
        :param e1: Ellipticity parameter 1.
        :type e1: float
        :param e2: Ellipticity parameter 2.
        :type e2: float
        :param center_x: x coordinate of centroid.
        :type center_x: float
        :param center_y: y coordinate of centroid.
        :type center_y: float
        :return: Convergence profile.
        :rtype: type(x)
        """
        amps, sigmas = self.gauss_decompose_sersic(n_sersic, R_sersic, k_eff)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gauss_decomposition.density_2d(x, y, amps, sigmas, e1, e2,
                                                   center_x, center_y)
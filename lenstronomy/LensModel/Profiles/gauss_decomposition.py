# -*- coding: utf-8 -*-
"""
This module constains the class to compute lensing properties of a set of
concentric elliptical Gaussian convergence profiles.
"""

__author__ = 'ajshajib'

import numpy as np
import abc
from scipy.special import comb
from future.utils import with_metaclass

from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import GaussianEllipseKappa
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil


class GaussianEllipseKappaSet(object):
    """
    This class computes the lensing properties of a set of concentric
    elliptical Gaussian convergences.
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'e1': 0.5, 'e2': 0.5,
                           'center_x': 100, 'center_y': 100}

    def __init__(self, use_scipy_wofz=True, min_ellipticity=1e-5):
        """

        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa(
        )``. If True, Gaussian lensing will use `scipy.special.wofz`
        function. Set False for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa(
        )``. Minimum ellipticity for Gaussian elliptical lensing calculation.
        For lower ellipticity than min_ellipticity the equations for the
        spherical case will be used.
        :type min_ellipticity: ``float``
        """
        self.gaussian_ellipse_kappa = GaussianEllipseKappa(
                                            use_scipy_wofz=use_scipy_wofz,
                                            min_ellipticity=min_ellipticity)

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the potential function for a set of concentric elliptical
        Gaussian convergence profiles.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian. Convention: A/(2*pi*sigma^2) *
        exp(-(x^2+y^2/q^2)/2/sigma^2).
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian.
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordianate of centroid.
        :type center_y: ``float``
        :return: Potential for elliptical Gaussian convergence.
        :rtype: ``float``, or ``numpy.array`` with ``shape = x.shape``.
        """
        function = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            function += self.gaussian_ellipse_kappa.function(x, y,
                                                             amp[i],
                                                             sigma[i], e1,
                                                             e2,
                                                             center_x,
                                                             center_y)
        return function

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the derivatives of function angles df/dx, df/dy at x,
        y for a set of concentric elliptic Gaussian convergence profiles.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian. Convention: A/(2*pi*sigma^2) *
        exp(-(x^2+y^2/q^2)/2/sigma^2).
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian.
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordianate of centroid.
        :type center_y: ``float``
        :return: Deflection angle df/dx, df/dy for elliptical Gaussian
        convergence.
        :rtype: tuple ``(float, float)`` or ``(numpy.array, numpy.array)``
        with each ``numpy`` array's shape equal to ``x.shape``.
        """
        f_x = np.zeros_like(x, dtype=float)
        f_y = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            f_x_i, f_y_i = self.gaussian_ellipse_kappa.derivatives(x, y,
                                                amp=amp[i],
                                                sigma=sigma[i], e1=e1,
                                                e2=e2, center_x=center_x,
                                                center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i

        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy for a
        set of concentric elliptic Gaussian convergence profiles.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian. Convention: A/(2*pi*sigma^2) *
        exp(-(x^2+y^2/q^2)/2/sigma^2).
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian.
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordianate of centroid.
        :type center_y: ``float``
        :return: Hessian d^2f/dx^2, d^f/dy^2, d^2/dxdy for elliptical
        Gaussian convergence.
        :rtype: tuple ``(float, float, float)`` , or ``(numpy.array, numpy.array,
        numpy.array)`` with each ``numpy`` array's shape equal to ``x.shape``.
        """
        f_xx = np.zeros_like(x, dtype=float)
        f_yy = np.zeros_like(x, dtype=float)
        f_xy = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_ellipse_kappa.hessian(
                                                            x, y,
                                                            amp=amp[i],
                                                            sigma=sigma[i],
                                                            e1=e1,
                                                            e2=e2,
                                                            center_x=center_x,
                                                            center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i

        return f_xx, f_yy, f_xy

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the density of a set of concentric elliptical Gaussian
        convergenc profiles \Sum {A/(2*pi*sigma^2) * exp(-(
        x^2+y^2/q^2)/2/sigma^2)}.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian. Convention: A/(2*pi*sigma^2) *
        exp(-(x^2+y^2/q^2)/2/sigma^2).
        :type amp: ``numpy.array`` with ``dtype=float``
        :param sigma: Standard deviation of Gaussian.
        :type sigma: ``numpy.array`` with ``dtype=float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordianate of centroid.
        :type center_y: ``float``
        :return: Density \kappa for elliptical
        Gaussian convergence.
        :rtype: ``float``, or ``numpy.array`` with shape equal to ``x.shape``.
        """
        density_2d = np.zeros_like(x, dtype=float)

        for i in range(len(amp)):
            density_2d += self.gaussian_ellipse_kappa.density_2d(x, y,
                                                    amp=amp[i],
                                                    sigma=sigma[i],
                                                    e1=e1, e2=e2,
                                                    center_x=center_x,
                                                    center_y=center_y)

        return density_2d


class GaussDecompositionAbstract(with_metaclass(abc.ABCMeta)):
    """
    This abstract class sets up a template for computing lensing properties of
    an elliptical convergence through Gaussian decomposition from Shajib
    (2019).
    """
    def __init__(self, n_sigma=15, sigma_start_mult=0.02, sigma_end_mult=15.,
                 precision=10, use_scipy_wofz=True, min_ellipticity=1e-5):
        """
        Set up settings for the Gaussian decomposition. For more details about
        the decomposition parameters, see Shajib (2019).

        :param n_sigma: Number of Gaussian components.
        :type n_sigma: ``int``
        :param sigma_start_mult: Lower range of logarithmically spaced sigmas.
        :type sigma_start_mult: ``float``
        :param sigma_end_mult: Upper range of logarithmically spaced sigmas.
        :type sigma_end_mult: ``float``
        :param precision: Numerical precision of Gaussian decomposition.
        :type precision: ``int``
        :param use_scipy_wofz: To be passed to ``class GaussianEllipseKappa(
        )``. If True, Gaussian lensing will use `scipy.special.wofz`
        function. Set False for lower precision, but faster speed.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: To be passed to ``class GaussianEllipseKappa(
        )``. Minimum ellipticity for Gaussian elliptical lensing calculation.
        For lower ellipticity than min_ellipticity the equations for the
        spherical case will be used.
        :type min_ellipticity: ``float``
        """
        self.gaussian_set = GaussianEllipseKappaSet(
                                            use_scipy_wofz=use_scipy_wofz,
                                            min_ellipticity=min_ellipticity)

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

    def gauss_decompose(self, **kwargs):
        r"""
        Compute the amplitudes and sigmas of Gaussian components using the
        integral transform with Gaussian kernel from Shajib (2019). The
        returned values are in the convention of eq. (2.13).

        :param func: the function to decompose.
        :type func: ``function``
        :param \*args: arguments to send to ``func``.
        :param \**kwargs: keyword arguments to send to ``func``.

        :return: Amplitudes and standard deviations of the Gaussian components.
        :rtype: tuple ``(numpy.array, numpy.array)``
        """
        sigma_start = self.sigma_start_mult*self.get_scale(**kwargs)
        sigma_end = self.sigma_end_mult*self.get_scale(**kwargs)

        sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_end),
                           self.n_sigma)

        f_sigmas = np.sum(self.etas * self.get_kappa_y(
                                sigmas[:,np.newaxis]*self.betas[np.newaxis, :],
                                **kwargs).real,
                          axis=1
                          )

        # weighting for trapezoid method integral
        f_sigmas[0] *= 0.5
        f_sigmas[-1] *= 0.5

        del_log_sigma = np.abs(np.diff(np.log(sigmas)).mean())

        f_sigmas *= del_log_sigma / np.sqrt(2.*np.pi)

        return f_sigmas, sigmas

    @abc.abstractmethod
    def get_scale(self, **kwargs):
        """
        Abstract method to identify the keyword argument for the get_scale size
        among the profile parameters of the child class' convergence profile.

        :param \**kwargs: Keyword arguments
        :return: Scale size
        :rtype: ``float``
        """
        pass

    @abc.abstractmethod
    def get_kappa_y(self, y, **kwargs):
        r"""
        Abstract method to compute the spherical Sersic profile at ``y``.
        The concrete method has to defined by the child class.

        :param y: y coordinate.
        :type y: float or numpy.array
        :param \**kwargs: Keyword arguments that are defined by the child
        class that are particular for the convergence profile in the child
        class.
        """
        pass

    def function(self, x, y, e1, e2, center_x=0.,
                 center_y=0., **kwargs):
        r"""
        Compute the deflection potential of a Gauss-decomposed
        elliptic convergence.

        :param x: x coordinate.
        :type x: ``float``
        :param y: y coordinate.
        :type y: ``float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordinate of centroid.
        :type center_y: ``float``
        :param \**kwargs: Keyword arguments that are defined by the child
        class that are particular for the convergence profile in the child
        class.
        :return: Deflection potential.
        :rtype: ``float``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2.*np.pi * sigmas * sigmas

        return self.gaussian_set.function(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def derivatives(self, x, y, e1, e2, center_x=0.,
                    center_y=0., **kwargs):
        r"""
        Compute the derivatives of the deflection potential df/dx, df/dy for a
        Gauss-decomposed elliptic convergence.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1.
        :type e1: float
        :param e2: Ellipticity parameter 2.
        :type e2: float
        :param center_x: x coordinate of centroid.
        :type center_x: float
        :param center_y: y coordinate of centroid.
        :type center_y: float
        :param \**kwargs: Keyword arguments that are defined by the child
        class that are particular for the convergence profile in the child
        class.
        :return: Derivatives of deflection potential.
        :rtype: tuple (type(x), type(x))
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gaussian_set.derivatives(x, y, amps, sigmas, e1, e2,
                                                    center_x, center_y)

    def hessian(self, x, y, e1, e2, center_x=0.,
                center_y=0., **kwargs):
        r"""
        Compute the Hessian of the deflection potential d^2f/dx^2,
        d^2f/dy^2, d^f/dxdy of a Gauss-decomposed elliptic Sersic convergence.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordinate of centroid.
        :type center_y: ``float``
        :param \**kwargs: Keyword arguments that are defined by the child
        class that are particular for the convergence profile in the child
        class.
        :return: Hessian of deflection potential.
        :rtype: tuple ``(type(x), type(x), type(x))``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gaussian_set.hessian(x, y, amps, sigmas, e1, e2,
                                                center_x, center_y)

    def density_2d(self, x, y, e1, e2, center_x=0.,
                   center_y=0., **kwargs):
        r"""
        Compute the convergence profile for Gauss-decomposed
        elliptic Sersic profile.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordinate of centroid.
        :type center_y: ``float``
        :param \**kwargs: Keyword arguments that are defined by the child
        class that are particular for the convergence profile in the child
        class.
        :return: Convergence profile.
        :rtype: ``type(x)``
        """
        amps, sigmas = self.gauss_decompose(**kwargs)

        # converting the amplitude convention A -> A/(2*pi*sigma^2)
        amps *= 2. * np.pi * sigmas * sigmas

        return self.gaussian_set.density_2d(x, y, amps, sigmas, e1, e2,
                                                   center_x, center_y)


class SersicEllipseGaussDec(GaussDecompositionAbstract):
    """
    This class computes the lensing properties of an elliptical Sersic
    profile using the Gauss decomposition method from Shajib (2019).
    """
    param_names = ['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x',
                   'center_y']
    lower_limit_default = {'k_eff': 0., 'R_sersic': 0., 'n_sersic': 0.5,
                           'e1': -0.5, 'e2': -0.5, 'center_x': -100.,
                           'center_y': -100.}
    upper_limit_default = {'k_eff': 100., 'R_sersic': 100., 'n_sersic': 8.,
                           'e1': 0.5, 'e2': 0.5, 'center_x': 100.,
                           'center_y': 100.}

    def get_kappa_y(self, y, **kwargs):
        r"""
        Compute the spherical Sersic profile at ``y``.

        :param y: y coordinate.
        :type y: ``float``
        :param \**kwargs: Keyword arguments

        :Keyword Arguments:
            * *n_sersic* (``float``) --
              Sersic index.
            * *R_sersic* (``float``) --
              Sersic get_scale radius.
            * *k_eff* (``float``) --
              Sersic convergence at R_sersic.

        :return: Sersic function at ``y``.
        :rtype: ``type(y)``
        """
        n_sersic = kwargs['n_sersic']
        R_sersic = kwargs['R_sersic']
        k_eff = kwargs['k_eff']

        bn = SersicUtil.b_n(n_sersic)

        return k_eff * np.exp(-bn * (y / R_sersic) ** (1. / n_sersic) + bn)

    def get_scale(self, **kwargs):
        """
        Identify the get_scale size from the keyword arguments.

        :param \**kwargs: Keyword arguments.
        :return: Scale size
        :rtype: ``float``
        """
        return kwargs['R_sersic']

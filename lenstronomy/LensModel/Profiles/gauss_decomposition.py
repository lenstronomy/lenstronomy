# -*- coding: utf-8 -*-
"""
This module constains the class to compute lensing properties of a set of
concentric elliptical Gaussian convergence profiles.
"""

__author__ = 'ajshajib'

import numpy as np

from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import GaussianEllipseKappa


class GaussDecomposition(object):
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
        self.gaussian_ellipse_kappa = GaussianEllipseKappa(
                                            use_scipy_wofz=use_scipy_wofz,
                                            min_ellipticity=min_ellipticity)

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :type x:
        :param y:
        :type y:
        :param amp: Array of amplitudes.
        :type amp: numpy array
        :param sigma: Array of sigmas.
        :type sigma: numpy array
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :param scale_factor:
        :type scale_factor:
        :return:
        :rtype:
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

        :param x:
        :type x:
        :param y:
        :type y:
        :param amp: array of amplitudes.
        :type amp: numpy array
        :param sigma: array of sigmas.
        :type sigma: numpy array
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :param scale_factor:
        :type scale_factor:
        :return:
        :rtype:
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

        :param x:
        :type x:
        :param y:
        :type y:
        :param amp: Array of amplitudes.
        :type amp: numpy array
        :param sigma: Array of sigmas.
        :type sigma: numpy array
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :param scale_factor:
        :type scale_factor:
        :return:
        :rtype:
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

        :param x:
        :type x:
        :param y:
        :type y:
        :param amp: Array of amplitudes.
        :type amp: numpy array
        :param sigma: Array of amplitudes.
        :type sigma: numpy array
        :param e1:
        :type e1:
        :param e2:
        :type e2:
        :param center_x:
        :type center_x:
        :param center_y:
        :type center_y:
        :param scale_factor:
        :type scale_factor:
        :return:
        :rtype:
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

# -*- coding: utf-8 -*-
"""
This module constains the class to compute lensing properties of a set of
concentric elliptical Gaussian convergence profiles.
"""

__author__ = 'ajshajib'

import numpy as np

from lenstronomy.LensModel.Profiles.gaussian_kappa_ellipse import GaussianKappaEllipse


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

    def __init__(self):
        self.gaussian_ellipse_kappa = GaussianKappaEllipse()

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
        n_dim = len(amp)

        is_scalar = False
        if np.isscalar(x) and np.isscalar(y):
            is_scalar = True
        is_naked_array = False
        if not x.shape and not y.shape:
            is_naked_array = True

        if np.isscalar(x) or not x.shape:
            x = np.array([x])
        if np.isscalar(y) or not y.shape:
            y = np.array([y])

        assert len(x) == len(y)

        xs = np.repeat(x[np.newaxis, :], n_dim, axis=0)
        ys = np.repeat(y[np.newaxis, :], n_dim, axis=0)

        amps = amp.reshape((-1,) + (1,) * (len(x.shape) ))
        sigmas = sigma.reshape((-1,) + (1,) * (len(x.shape)))

        f_x, f_y = self.gaussian_ellipse_kappa.derivatives(xs, ys,
                                                amp=amps,
                                                sigma=sigmas, e1=e1,
                                                e2=e2, center_x=center_x,
                                                center_y=center_y)

        f_x, f_y = np.sum(f_x, axis=0), np.sum(f_y, axis=0)

        if is_scalar:
            return f_x[0], f_y[0]
        elif is_naked_array:
            return np.reshape(f_x, ()), np.reshape(f_y, ())
        else:
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
        n_dim = len(amp)

        is_scalar = False
        if np.isscalar(x) and np.isscalar(y):
            is_scalar = True
        is_naked_array = False
        if not x.shape and not y.shape:
            is_naked_array = True

        if np.isscalar(x) or not x.shape:
            x = np.array([x])
        if np.isscalar(y) or not y.shape:
            y = np.array([y])
        assert len(x) == len(y)

        xs = np.repeat(x[np.newaxis, :], n_dim, axis=0)
        ys = np.repeat(y[np.newaxis, :], n_dim, axis=0)

        amps = amp.reshape((-1,) + (1,) * (len(x.shape)))
        sigmas = sigma.reshape((-1,) + (1,) * (len(x.shape)))

        f_xx, f_yy, f_xy = self.gaussian_ellipse_kappa.hessian(xs, ys,
                                                amp=amps,
                                                sigma=sigmas, e1=e1,
                                                e2=e2, center_x=center_x,
                                                center_y=center_y)
        f_xx, f_yy, f_xy = np.sum(f_xx, axis=0), np.sum(f_yy, axis=0), np.sum(
            f_xy, axis=0)

        if is_scalar:
            return f_xx[0], f_yy[0], f_xy[0]
        elif is_naked_array:
            return np.reshape(f_xx, ()), np.reshape(f_yy, ()), np.reshape(
                f_xy, ())
        else:
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
        n_dim = len(amp)

        is_scalar = False
        if np.isscalar(x) and np.isscalar(y):
            is_scalar = True
        is_naked_array = False
        if not x.shape and not y.shape:
            is_naked_array = True

        if np.isscalar(x) or not x.shape:
            x = np.array([x])
        if np.isscalar(y) or not y.shape:
            y = np.array([y])
        assert len(x) == len(y)

        xs = np.repeat(x[np.newaxis, :], n_dim, axis=0)
        ys = np.repeat(y[np.newaxis, :], n_dim, axis=0)

        amps = amp.reshape((-1,) + (1,) * (len(x.shape)))
        sigmas = sigma.reshape((-1,) + (1,) * (len(x.shape)))

        density_2d = self.gaussian_ellipse_kappa.density_2d(xs, ys,
                                                    amp=amps,
                                                    sigma=sigmas,
                                                    e1=e1, e2=e2,
                                                    center_x=center_x,
                                                    center_y=center_y)
        density_2d = np.sum(density_2d, axis=0)

        if is_scalar:
            return density_2d[0]
        elif is_naked_array:
            return np.reshape(density_2d, ())
        else:
            return density_2d

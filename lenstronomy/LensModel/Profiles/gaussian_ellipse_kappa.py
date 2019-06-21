# -*- coding: utf-8 -*-
"""
This module defines ``class GaussianEllipseKappa`` to compute the lensing
properties of an elliptical Gaussian profile with ellipticity in the
convergence using the formulae from Shajib (2019).
"""

__author__ = 'ajshajib'

import numpy as np
from scipy.special import wofz
from scipy.integrate import quad
from copy import deepcopy
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import lenstronomy.Util.param_util as param_util


class GaussianEllipseKappa(object):
    """
    This class contains functions to evaluate the derivative and hessian matrix
    of the deflection potential for an elliptical Gaussian convergence.

    The formulae are from Shajib (2019).
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'e1': 0.5, 'e2': 0.5,
                           'center_x': 100, 'center_y': 100}

    def __init__(self, use_scipy_wofz=True, min_ellipticity=1e-5):
        """
        Setup which method to use the Faddeeva function and the
        ellipticity limit for spherical approximation.

        :param use_scipy_wofz: If ``True``, use ``scipy.special.wofz``.
        :type use_scipy_wofz: ``bool``
        :param min_ellipticity: Minimum allowed ellipticity. For ``q > 1 - min_ellipticity``, values for spherical case will be returned.
        :type min_ellipticity: ``float``
        """
        if use_scipy_wofz:
            self.w_f = wofz
        else:
            self.w_f = self.w_f_approx

        self.min_ellipticity = min_ellipticity
        self.spherical = GaussianKappa()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the potential function for elliptical Gaussian convergence.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \pi\sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)`
        :type amp: ``float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :return: Potential for elliptical Gaussian convergence
        :rtype: ``float``, or ``numpy.array`` with shape equal to ``x.shape``
        """
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)

        if q > 1 - self.min_ellipticity:
            return self.spherical.function(x, y, amp, sigma, center_x,
                                           center_y)

        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * np.pi * sigma**2)

        # converting ellipticity definition from x^2 + y^2/q^2 to q^2*x^2 + y^2
        sigma_ = sigma * q

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_g)
        sin_phi = np.sin(phi_g)

        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        _b = 1. / 2. / sigma_ ** 2
        _p = np.sqrt(_b * q ** 2 / (1. - q ** 2))

        def pot_real_line_integrand(_x):
            sig_func_re, sig_func_im = self.sigma_function(_p * _x, 0, q)

            alpha_x_ = amp_*sigma_ * self.sgn(_x) * np.sqrt(2*np.pi / (
                    1. - q ** 2)) * sig_func_re

            return alpha_x_

        def pot_imag_line_integrand(_y):
            sig_func_re, sig_func_im = self.sigma_function(_p * x_, _p * _y, q)

            alpha_y_ = -amp_*sigma_ * self.sgn(x_ + 1j*_y) * np.sqrt(2*np.pi /
                        (1. - q ** 2)) * sig_func_im

            return alpha_y_

        pot_on_real_line = quad(pot_real_line_integrand, 0, x_)[0]
        pot_on_imag_parallel = quad(pot_imag_line_integrand, 0, y_)[0]

        return (pot_on_real_line + pot_on_imag_parallel)

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the derivatives of function angles :math:`\partial
        f/\partial x`, :math:`\partial f/\partial y` at :math:`x,\ y`.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \pi\sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)`
        :type amp: ``float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :return: Deflection angle :math:`\partial f/\partial x`, :math:`\partial f/\partial y` for elliptical Gaussian convergence.
        :rtype: tuple ``(float, float)`` or ``(numpy.array, numpy.array)`` with each ``numpy.array``'s shape equal to ``x.shape``.
        """
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)

        if q > 1 - self.min_ellipticity:
            return self.spherical.derivatives(x, y, amp, sigma, center_x,
                                              center_y)

        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * np.pi * sigma**2)

        # converting ellipticity definition from x^2 + y^2/q^2 to q^2*x^2 + y^2
        sigma_ = sigma * q

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_g)
        sin_phi = np.sin(phi_g)

        # rotated coordinates
        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        _p = q / sigma_ / np.sqrt(2 * (1. - q**2))

        sig_func_re, sig_func_im = self.sigma_function(_p * x_, _p * y_, q)

        alpha_x_ = amp_ * sigma_ * self.sgn(x_ + 1j*y_) * np.sqrt(2*np.pi/(
                1.-q**2)) * sig_func_re
        alpha_y_ = - amp_ * sigma_ * self.sgn(x_ + 1j*y_) * np.sqrt(
            2 * np.pi / (1. - q ** 2)) * sig_func_im

        # rotate back to the original frame
        f_x = alpha_x_ * cos_phi - alpha_y_ * sin_phi
        f_y = alpha_x_ * sin_phi + alpha_y_ * cos_phi

        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute Hessian matrix of function :math:`\partial^2f/\partial x^2`,
        :math:`\partial^2 f/\partial y^2`, :math:`\partial^2/\partial
        x\partial y`.

        :param x: x coordinate
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \pi\sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)`
        :type amp: ``float``
        :param sigma: Standard deviation of Gaussian
        :type sigma: ``float``
        :param e1: Ellipticity parameter 1
        :type e1: ``float``
        :param e2: Ellipticity parameter 2
        :type e2: ``float``
        :param center_x: x coordinate of centroid
        :type center_x: ``float``
        :param center_y: y coordianate of centroid
        :type center_y: ``float``
        :return: Hessian :math:`A/(2 \pi \sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)` for elliptical Gaussian convergence.
        :rtype: tuple ``(float, float, float)`` , or ``(numpy.array, numpy.array, numpy.array)`` with each ``numpy.array``'s shape equal to ``x.shape``.
        """
        phi_g, q = param_util.ellipticity2phi_q(e1, e2)

        if q > 1 - self.min_ellipticity:
            return self.spherical.hessian(x, y, amp, sigma, center_x, center_y)

        # adjusting amplitude to make the notation compatible with the
        # formulae given in Shajib (2019).
        amp_ = amp / (2 * np.pi * sigma**2)

        # converting ellipticity definition from x^2 + y^2/q^2 to q^2*x^2 + y^2
        sigma_ = sigma * q

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_g)
        sin_phi = np.sin(phi_g)

        # rotated coordinates
        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        _p = q / sigma_ / np.sqrt(2 * (1. - q**2))
        sig_func_re, sig_func_im = self.sigma_function(_p * x_, _p * y_, q)

        kappa = amp_ * np.exp(-(q**2 * x_**2 + y_**2) / 2 / sigma_**2)

        shear = - 1/(1-q*q) * ((1+q**2)*kappa - 2*q*amp_ + np.sqrt(
            2*np.pi) * q*q * amp_ * (x_ - 1j*y_) / sigma_ / np.sqrt(1-q*q) * (
            sig_func_re - 1j*sig_func_im))

        # in rotated frame
        f_xx_ = kappa + shear.real
        f_yy_ = kappa - shear.real
        f_xy_ = shear.imag

        # rotate back to the original frame
        f_xx = f_xx_ * cos_phi**2 + f_yy_ * sin_phi**2 \
               - 2 * sin_phi * cos_phi * f_xy_
        f_yy = f_xx_ * sin_phi**2 + f_yy_ * cos_phi**2 \
               + 2 * sin_phi * cos_phi * f_xy_
        f_xy = sin_phi * cos_phi * (f_xx_ - f_yy_) \
               + (cos_phi**2 - sin_phi**2) * f_xy_

        return f_xx, f_yy, f_xy

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the density of elliptical Gaussian :math:`A/(2 \pi
        \sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)`.

        :param x: x coordinate.
        :type x: ``float`` or ``numpy.array``
        :param y: y coordinate.
        :type y: ``float`` or ``numpy.array``
        :param amp: Amplitude of Gaussian, convention: :math:`A/(2 \pi\sigma^2) \exp(-(x^2+y^2/q^2)/2\sigma^2)`
        :type amp: ``float``
        :param sigma: Standard deviation of Gaussian.
        :type sigma: ``float``
        :param e1: Ellipticity parameter 1.
        :type e1: ``float``
        :param e2: Ellipticity parameter 2.
        :type e2: ``float``
        :param center_x: x coordinate of centroid.
        :type center_x: ``float``
        :param center_y: y coordianate of centroid.
        :type center_y: ``float``
        :return: Density :math:`\kappa` for elliptical Gaussian convergence.
        :rtype: ``float``, or ``numpy.array`` with shape = ``x.shape``.
        """
        f_xx, f_yy, f_xy = self.hessian(x, y, amp, sigma, e1, e2, center_x,
                                        center_y)
        return (f_xx + f_yy) / 2

    @staticmethod
    def sgn(z):
        """
        Compute the sign function :math:`\mathrm{sgn}(z)` factor for
        deflection as sugggested by Bray (1984). For current implementation, returning 1 is sufficient.

        :param z: Complex variable :math:`z = x + \mathrm{i}y`
        :type z: ``complex``
        :return: :math:`\mathrm{sgn}(z)`
        :rtype: ``float``
        """
        return 1.
        # np.sqrt(z*z)/z #np.sign(z.real*z.imag)
        #return np.sign(z.real)
        #if z.real != 0:
        #    return np.sign(z.real)
        #else:
        #    return np.sign(z.imag)
        #return np.where(z.real == 0, np.sign(z.real), np.sign(z.imag))

    def sigma_function(self, x, y, q):
        r"""
        Compute the function :math:`\varsigma (z; q)` from equation (4.12)
        of Shajib (2019).

        :param x: Real part of complex variable, :math:`x = \mathrm{Re}(z)`
        :type x: ``float`` or ``numpy.array``
        :param y: Imaginary part of complex variable, :math:`y = \mathrm{Im}(z)`
        :type y: ``float`` or ``numpy.array``
        :param q: Axis ratio
        :type q: ``float``
        :return: real and imaginary part of :math:`\varsigma(z; q)` function
        :rtype: tuple ``(type(x), type(x))``
        """
        y_sign = np.sign(y)
        y_ = deepcopy(y) * y_sign
        z = x + 1j * y_
        zq = q * x + 1j * y_ / q

        w = self.w_f(z)
        wq = self.w_f(zq)

        # exponential factor in the 2nd term of eqn. (4.15) of Shajib (2019)
        exp_factor = np.exp(-x * x * (1 - q * q) - y_ * y_ * (1 / q / q - 1))

        sigma_func_real = w.imag - exp_factor * wq.imag
        sigma_func_imag = (- w.real + exp_factor * wq.real) * y_sign

        return sigma_func_real, sigma_func_imag

    @staticmethod
    def w_f_approx(z):
        """
        Compute the Faddeeva function :math:`w_{\mathrm F}(z)` using the
        approximation given in Zaghloul (2017).

        :param z: complex number
        :type z: ``complex`` or ``numpy.array(dtype=complex)``
        :return: :math:`w_\mathrm{F}(z)`
        :rtype: ``complex``
        """
        sqrt_pi = 1 / np.sqrt(np.pi)
        i_sqrt_pi = 1j * sqrt_pi

        wz = np.empty_like(z)

        z_imag2 = z.imag ** 2
        abs_z2 = z.real ** 2 + z_imag2

        reg1 = (abs_z2 >= 38000.)
        if np.any(reg1):
            wz[reg1] = i_sqrt_pi / z[reg1]

        reg2 = (256. <= abs_z2) & (abs_z2 < 38000.)
        if np.any(reg2):
            t = z[reg2]
            wz[reg2] = i_sqrt_pi * t / (t * t - 0.5)

        reg3 = (62. <= abs_z2) & (abs_z2 < 256.)
        if np.any(reg3):
            t = z[reg3]
            wz[reg3] = (i_sqrt_pi / t) * (1 + 0.5 / (t * t - 1.5))

        reg4 = (30. <= abs_z2) & (abs_z2 < 62.) & (z_imag2 >= 1e-13)
        if np.any(reg4):
            t = z[reg4]
            tt = t * t
            wz[reg4] = (i_sqrt_pi * t) * (tt - 2.5) / (tt * (tt - 3.) + 0.75)

        reg5 = (62. > abs_z2) & np.logical_not(reg4) & (abs_z2 > 2.5) & (
                    z_imag2 < 0.072)
        if np.any(reg5):
            t = z[reg5]
            u = -t * t
            f1 = sqrt_pi
            f2 = 1
            s1 = [1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31]
            s2 = [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84,
                  32066.6]

            for s in s1:
                f1 = s - f1 * u
            for s in s2:
                f2 = s - f2 * u

            wz[reg5] = np.exp(u) + 1j * t * f1 / f2

        reg6 = (30.0 > abs_z2) & np.logical_not(reg5)
        if np.any(reg6):
            t3 = - 1j * z[reg6]

            f1 = sqrt_pi
            f2 = 1
            s1 = [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239,
                  122.60793]
            s2 = [10.479857, 53.992907, 170.35400, 348.70392, 457.33448,
                  352.73063, 122.60793]

            for s in s1:
                f1 = f1 * t3 + s
            for s in s2:
                f2 = f2 * t3 + s

            wz[reg6] = f1 / f2
        return wz

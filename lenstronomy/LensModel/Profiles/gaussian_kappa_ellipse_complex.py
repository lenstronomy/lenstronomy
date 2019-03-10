__author__ = 'ajshajib'
#this file contains a class to make a gaussian

import numpy as np
from scipy.special import erfcx
from scipy.special import erf
from scipy.special import wofz
from copy import deepcopy
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import lenstronomy.Util.param_util as param_util
#from mpmath import erfi

class GaussianKappaEllipse(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    with ellipticity in the convergence

    the equations are derived using complex formulation of lensing following Shajib 2019.

    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        #self.spherical = GaussianKappa()
        #self._diff = 0.000001
        pass

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        raise('Not implemented yet!')

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)

        return -1

    def deflection(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Compute the deflection alpha_x alpha_y at x, y.
        :param x:
        :type x:
        :param y:
        :type y:
        :param amp:
        :type amp:
        :param sigma:
        :type sigma:
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
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        #e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) #* np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) # * np.sqrt(1 + e)

        _b = 1. / 2. / sigma**2
        _p = np.sqrt(_b * q**2 / (1. - q**2))

        #print(_p * x, _p * y / q)

        #if not np.isscalar(x):
        #   x_[np.abs(_p*x_)>26.] = 0.
        #    y_[np.abs(_p*y_/q)>26.] = 0.
        #elif np.abs(_p*x_)>26. or np.abs(_p*y_/q)>26.:
        #    x_ = 0.
        #    y_ = 0.

        #derfi = erfi(_p * (x_ + 1j*y_)) - erfi(_p*(q*x_ + 1j*y_/q))
        ddaw_er, ddaw_ei = self.ddaw_elliptical(_p * x_, _p * y_, q)

        #print(_p * x, _p * y / q, derfi)
        #print(x_, y_, amp, q, sigma, derfi)
        #derfi = np.float128(derfi.tolist())
        alpha_real = amp * sigma * self.sgn(x_+1j*y_) * np.sqrt(2*np.pi/(
                1.-q**2)) * ddaw_er
        alpha_imag = -amp * sigma * self.sgn(x_ + 1j * y_) * np.sqrt(
            2 * np.pi / (
                    1. - q ** 2)) * ddaw_ei

        #if np.isnan(alpha.any()) or np.isinf(alpha.any()):
        #print(x_, y_, amp, q, sigma, alpha)
        #print(_b, _p, q, x_, y_, alpha, _p * (x_ + 1j * y_), erfi(_p * (x_ +
        #                                                               1j *
        #
        #                                                                y_)),  _p * (q * x_ + 1j * y_ / q), erfi(
        #                _p * (q * x_ + 1j * y_ / q)))

        #if np.isnan(alpha.real).any():
        #    print(q, x_[np.isnan(alpha.real)], y_[np.isnan(alpha.real)])

        return alpha_real, alpha_imag

    def shear(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Return the shear gamma_1 and gamma_2 at x, y.
        :param x:
        :type x:
        :param y:
        :type y:
        :param amp:
        :type amp:
        :param sigma:
        :type sigma:
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
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        #e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) #* np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) #* np.sqrt(1 + e)

        q2 = 1. - q**2
        rq2 = np.sqrt(q2)
        p = np.sqrt(np.pi)
        s2 = sigma ** 2
        s = sigma
        r2 = np.sqrt(2.)

        _bb = 1. / 2. / sigma ** 2
        _pp = np.sqrt(1 / 2 / (1. - q**2)) * q / sigma

        ddaw_er, ddaw_ei = self.ddaw_elliptical(_pp * x_, _pp * y_, q)

        #print(np.exp(-(q**2 * x_**2 + y_**2)), x_, y_, q)

        shear_real =  - amp / (q2**1.5 * s) * (s * rq2 * ((1. + q**2) * np.exp(
            -(q**2 * x_**2 + y_**2) / (2. * s2)) - 2. * q) + r2 * p * q**2
            * (x_* ddaw_er - y_ * ddaw_ei ))

        shear_imag =  amp / (q2**1.5 * s) * (r2 * p * q**2
            * (x_ * ddaw_ei + y_ * ddaw_er ))

        return shear_real, shear_imag

    def kappa(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        Return the convergence at x, y.
        :param x:
        :type x:
        :param y:
        :type y:
        :param amp:
        :type amp:
        :param sigma:
        :type sigma:
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
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) #* np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) #* np.sqrt(1 + e)

        return amp * np.exp(-(q**2*x_**2+y_**2)/2/sigma**2)

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        f_x, f_y = self.deflection(x, y, amp, sigma, e1, e2, center_x,
                                   center_y)
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        if x == [] or y == []:
            return [], [], []

        kappa = self.kappa(x, y, amp, sigma, e1, e2, center_x, center_y)
        #print(x, kappa)
        g1, g2 = self.shear(x, y, amp, sigma, e1, e2, center_x, center_y)

        f_xx = kappa + g1
        f_yy = kappa - g1
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_xy = g2
        #print(x, y, f_xx, f_yy, f_xy)
        return f_xx, f_yy, f_xy

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return self.kappa(x, y, amp, sigma, e1, e2, center_x, center_y)

    @staticmethod
    def sgn(z):
        return 1.  # np.sqrt(z*z)/z #np.sign(z.real*z.imag)
        #return np.sign(z.real)
        #if z.real != 0:
        #    return np.sign(z.real)
        #else:
        #    return np.sign(z.imag)
        #return np.where(z.real == 0, np.sign(z.real), np.sign(z.imag))

    @staticmethod
    def ddaw_elliptical(x, y, q):
        y_sign = np.sign(y)
        y_ = deepcopy(y) * y_sign
        z = x + 1j * y_
        zq = q * x + 1j * y_ / q

        w = wofz(z)
        wq = wofz(zq)

        expxyqm1 = np.exp(-x * x * (1 - q * q) - y_ * y_ * (1 / q / q - 1))

        ddaw_real = w.imag - expxyqm1 * wq.imag
        ddaw_imag = (- w.real + expxyqm1 * wq.real) * y_sign

        return ddaw_real, ddaw_imag

    @staticmethod
    def ddaw_elliptical_old(x, y, q, abs_tol=1e-10, rel_tol=1e-10):
        """
        Compute the elliptical difference in Dawson function at z=x+iy.
        """
        if x is [] or y is []:
            return []

        a = np.pi / np.sqrt(-np.log(abs_tol * 0.5))  # 0.5
        # print(a)

        y_sign = np.sign(y)
        y *= y_sign

        cos2xy = np.cos(2 * x * y)
        sin2xy = np.sin(2 * x * y)
        sinxy = np.sin(x * y)

        if np.isscalar(x * y):
            if x * y == 0:
                sin2xydxy = 1
                sinxydxy = 1
            else:
                sin2xydxy = sin2xy / x / y
                sinxydxy = sinxy / x / y
        else:
            sin2xydxy = np.zeros_like(x * y)
            sinxydxy = np.zeros_like(x * y)
            sin2xydxy[x * y == 0] = 1
            sin2xydxy[x * y != 0] = sin2xy[x * y != 0] / (x * y)[x * y != 0]
            sinxydxy[x * y == 0] = 1
            sinxydxy[x * y != 0] = sinxy[x * y != 0] / (x * y)[x * y != 0]

        derfcx = - erfcx(y) + np.exp(-y * y * (1 / q / q - 1)) * erfcx(y / q)

        expxx = np.exp(-x * x)
        expqxy2 = np.exp(-x * x * (1 - q * q) - y * y * (1 / q / q - 1))

        real_ddaw = expxx * sin2xy * (
            derfcx) + expxx * 2 * a * x * sin2xydxy / np.pi * (
                                1 - q * np.exp(-y * y * (1 / q / q - 1)))
        imag_ddaw = expxx * cos2xy * (
            derfcx) - expxx * 2 * a * x * sinxy * sinxydxy / np.pi * (
                                1 - q * np.exp(-y * y * (1 / q / q - 1)))

        n = 1
        if np.isscalar(x):
            N = 20 + int(np.ceil(np.abs(x / a)))
        else:
            try:
            #print(len(x), x.shape, x)
                N = 20 + int(np.ceil(np.max(np.abs(x / a).flatten())))
            except ValueError:
                N = 20
        # print(N)
        while n < N:
            s1 = np.exp(-a * a * n * n - x * x) / (a * a * n * n + y * y)
            s1q = np.exp(-a * a * n * n - x * x * q * q) / (
                        a * a * n * n + y * y / q / q) * expqxy2

            s2 = np.exp(-(a * n + x) * (a * n + x)) / (a * a * n * n + y * y)
            s2q = np.exp(-(a * n + q * x) * (a * n + q * x)) / (
                        a * a * n * n + y * y / q / q) * expqxy2

            s3 = np.exp(-(a * n - x) * (a * n - x)) / (a * a * n * n + y * y)
            s3q = np.exp(-(a * n - q * x) * (a * n - q * x)) / (
                        a * a * n * n + y * y / q / q) * expqxy2

            s4 = s2 * a * n
            s4q = s2q * a * n

            s5 = s3 * a * n
            s5q = s3q * a * n

            real_delta = 2 * a / np.pi * (y * sin2xy * (s1 - s1q / q) - 0.5 * (
                        s4 - s5 - s4q + s5q))
            imag_delta = 2 * a / np.pi * (
                        y * cos2xy * (s1 - s1q / q) - y / 2 * (
                            s2 + s3 - s2q / q - s3q / q))

            real_ddaw += real_delta
            imag_ddaw += imag_delta

            n += 1

        #if np.isnan(real_ddaw).any():
        #    print(q, x[np.isnan(real_ddaw)], y[np.isnan(real_ddaw)])

        #if np.isnan(imag_ddaw).any():
        #    print(q, x[np.isnan(imag_ddaw)], y[np.isnan(imag_ddaw)])

        y *= y_sign

        return real_ddaw, imag_ddaw*y_sign

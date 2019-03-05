__author__ = 'ajshajib'
#this file contains a class to make a gaussian

import numpy as np
from scipy.special import erfi
from scipy.special import erf
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import lenstronomy.Util.param_util as param_util


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
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) #* np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) # * np.sqrt(1 + e)

        _b = 1 / 2 / sigma ** 2
        _p = np.sqrt(_b * q ** 2 / (1 - q ** 2))

        alpha = amp * self.sgn(x_ + 1j * y_) * np.sqrt(
            np.pi / _b / (1 - q ** 2)) * np.exp(-_p ** 2 * (x_ + 1j * y_) **
                                                2) * (
                            erfi(_p * (x_ + 1j * y_)) - erfi(
                        _p * (q * x_ + 1j * y_ / q)))

        return alpha.real, -alpha.imag

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
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) #* np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) #* np.sqrt(1 + e)

        q2 = 1 - q ** 2
        rq2 = np.sqrt(q2)
        p = np.sqrt(np.pi)
        s2 = sigma ** 2
        s = sigma
        r2 = np.sqrt(2)

        shear = - amp / (q2**1.5 * s) * (s * rq2 * ((1 + q**2) * np.exp(
            -(q**2 * x_**2 + y_**2) / (2 * s2)) - 2 * q) + r2 * p * q**2
            * (x_ + 1j*y_) * np.exp(-q**2 * (x_ + 1j*y_)**2 / 2 / q2 / s2)
            * (erfi(q * (x_ + 1j*y_) / r2 / rq2 / s) - erfi((q**2 * x_
                                                    + 1j*y_) / r2 / rq2 / s)))

        return shear.real, -shear.imag

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
        g1, g2 = self.shear(x, y, amp, sigma, e1, e2, center_x, center_y)
        kappa = self.convergence(x, y, amp, sigma, e1, e2, center_x, center_y)

        f_xx = kappa + g1
        f_yy = kappa - g1
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_xy = g2

        return f_xx, f_yy, f_xy

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return self.convergence(x, y, amp, sigma, e1, e2, center_x, center_y)

    @staticmethod
    def sgn(z):
        return 1.  # np.sqrt(z*z)/z #np.sign(z.real*z.imag)
        if z.real != 0:
            return np.sign(z.real)
        else:
            return np.sign(z.imag)
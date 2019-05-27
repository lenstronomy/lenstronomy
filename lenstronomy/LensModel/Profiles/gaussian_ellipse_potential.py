__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
import lenstronomy.Util.param_util as param_util


class GaussianEllipsePotential(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    with ellipticity in the convergence

    the calculation follows Glenn van de Ven et al. 2009

    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.spherical = GaussianKappa()
        self._diff = 0.000001

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns Gaussian
        """

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
        f_ = self.spherical.function(x_, y_, amp=amp, sigma=sigma)
        return f_

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)

        f_x_prim, f_y_prim = self.spherical.derivatives(x_, y_, amp=amp, sigma=sigma)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, amp, sigma, e1, e2, center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, amp, sigma, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, amp, sigma, e1, e2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_yy, f_xy

    def density(self, r, amp, sigma, e1, e2):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        return self.spherical.density(r, amp, sigma)

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return self.spherical.density_2d(x, y, amp, sigma, center_x, center_y)

    def mass_2d(self, R, amp, sigma, e1, e2):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return self.spherical.mass_2d(R, amp, sigma)

    def mass_3d(self, R, amp, sigma, e1, e2):
        """

        :param R:
        :param amp:
        :param sigma:
        :param e1:
        :param e2:
        :return:
        """
        return self.spherical.mass_3d(R, amp, sigma)

    def mass_3d_lens(self, R, amp, sigma, e1, e2):
        """

        :param R:
        :param amp:
        :param sigma:
        :param e1:
        :param e2:
        :return:
        """
        return self.spherical.mass_3d_lens(R, amp, sigma)

    def mass_2d_lens(self, R, amp, sigma, e1, e2):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return self.spherical.mass_2d_lens(R, amp, sigma)

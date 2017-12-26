__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
from lenstronomy.LensModel.Profiles.sersic import Sersic


class SersicEllipse(object):
    """
    this class contains functions to evaluate a Sersic mass profile: https://arxiv.org/pdf/astro-ph/0311559.pdf
    """
    def __init__(self):
        self.sersic = Sersic()
        self._diff = 0.000001

    def function(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        f_ = self.sersic.function(x_, y_, n_sersic, r_eff, k_eff)
        return f_

    def derivatives(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        e = abs(1. - q)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        f_x_prim, f_y_prim = self.sersic.derivatives(x_, y_, n_sersic, r_eff, k_eff)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff

        return f_xx, f_yy, f_xy

    def _coord_transf(self, x, y, q, phi_G, center_x, center_y):
        """

        :param x:
        :param y:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
        return x_, y_
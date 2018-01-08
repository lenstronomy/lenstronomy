__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_ellipse import SersicEllipse


class SersicDouble(object):
    """
    this class contains functions to evaluate a Sersic mass profile: https://arxiv.org/pdf/astro-ph/0311559.pdf
    """
    def __init__(self):
        self.sersic = SersicEllipse()
        self._diff = 0.000001

    def function(self, x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        k_eff2 = k_eff * flux_ratio
        f_1 = self.sersic.function(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        f_2 = self.sersic.function(x, y, n_2, R_2, k_eff2, q_2, phi_G_2, center_x, center_y)
        return f_1 + f_2

    def derivatives(self, x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        k_eff2 = k_eff * flux_ratio
        f_x1, f_y1 = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        f_x2, f_y2 = self.sersic.derivatives(x, y, n_2, R_2, k_eff2, q_2, phi_G_2, center_x, center_y)
        return f_x1 + f_x2, f_y1 + f_y2

    def hessian(self, x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        k_eff2 = k_eff * flux_ratio
        f_xx1, f_yy1, f_xy1 = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff, q, phi_G, center_x, center_y)
        f_xx2, f_yy2, f_xy2 = self.sersic.hessian(x, y, n_2, R_2, k_eff2, q_2, phi_G_2, center_x, center_y)
        return f_xx1 + f_xx2, f_yy1 + f_yy2, f_xy1 + f_xy2

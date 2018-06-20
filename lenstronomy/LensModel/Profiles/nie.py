__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.param_util as param_util


class NIE(object):
    """
    this class contains the function and the derivatives of the  (softened) Isothermal ellipse
    See Keeton&Kochanek 1998
    """
    def function(self, x, y, theta_E, smoothing_scale, e1, e2, center_x=0, center_y=0):
        x_shift = x - center_x
        y_shift = y - center_y
        f_ = theta_E * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        return f_

    def derivatives(self, x, y, theta_E, smoothing_scale, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        alpha = theta_E / (1 - q ** 2)
        #return f_x, f_y
        return 0, 0

    def hessian(self, x, y, theta_E, smoothing_scale, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        alpha = theta_E / ( 1 - q**2)
        #return f_xx, f_yy, f_xy
        return 0, 0, 0

    def _psi(self, x, y, q, s):
        """
        expression after equation (8) in Keeton&Kochanek 1998

        :param x:
        :param y:
        :param q:
        :param s:
        :return:
        """
        return np.sqrt(q**2 * (s**2 + x**2) + y**2)
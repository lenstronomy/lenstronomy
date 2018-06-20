__author__ = 'sibirrer'

import numpy as np


class NIE_simple(object):
    """
    this class contains the function and the derivatives of the  (softened) Isothermal ellipse
    See Keeton&Kochanek 1998
    """
    def __init__(self, diff=0.00000001):
        self._diff = diff

    def function(self, x, y, theta_E, s, q):
        psi = self._psi(x, y, q, s)
        alpha_x, alpha_y = self.derivatives(x, y, theta_E, s, q)
        f_ = x*alpha_x + y*alpha_y - theta_E * s * 1./2. * np.log((psi+s)**2 + (1. - q**2)*x**2)
        return f_

    def derivatives(self, x, y, theta_E, s, q):
        """
        returns df/dx and df/dy of the function
        """
        psi = self._psi(x, y, q, s)
        f_x = theta_E / np.sqrt(1. -q**2) * np.arctan(np.sqrt((1.-q**2))*x / (psi+s))
        f_y = theta_E / np.sqrt(1. - q ** 2) * np.arctan(np.sqrt((1. - q ** 2)) * y / (psi + q**2*s))
        return f_x, f_y

    def hessian(self, x, y, theta_E, s, q):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, theta_E, s, q)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, theta_E, s, q)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, theta_E, s, q)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        # f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_yy, f_xy

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
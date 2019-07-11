__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util


class NIE(object):
    """

    """
    param_names = ['theta_E', 'e1', 'e2', 's_scale', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 's_scale': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 10, 'e1': 0.5, 'e2': -0.5, 's_scale': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.nie_simple = NIE_simple()

    def function(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param e1:
        :param e2:
        :param s_scale:
        :param center_x:
        :param center_y:
        :return:
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E = self._theta_E_q_convert(theta_E, q)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self.nie_simple.function(x__, y__, theta_E, s_scale, q)
        # rotate back
        return f_

    def derivatives(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param e1:
        :param e2:
        :param s_scale:
        :param center_x:
        :param center_y:
        :return:
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E = self._theta_E_q_convert(theta_E, q)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__x, f__y = self.nie_simple.derivatives(x__, y__, theta_E, s_scale, q)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -phi_G)
        return f_x, f_y

    def hessian(self, x, y, theta_E, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param e1:
        :param e2:
        :param s_scale:
        :param center_x:
        :param center_y:
        :return:
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        theta_E = self._theta_E_q_convert(theta_E, q)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f__xx, f__yy, f__xy = self.nie_simple.hessian(x__, y__, theta_E, s_scale, q)
        # rotate back
        kappa = 1./2 * (f__xx + f__yy)
        gamma1__ = 1./2 *(f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def _theta_E_q_convert(self, theta_E, q):
        """
        converts a spherical averaged Einstein radius to an elliptical (major axis) Einstein radius.
        This then follows the convention of the SPEMD profile in lenstronomy.
        (theta_E / theta_E_gravlens) = sqrt[ (1+q^2) / (2 q) ]

        :param theta_E:
        :param q:
        :return:
        """
        #theta_E_new = theta_E / (np.sqrt((1.+q**2) / (2. * q))) /(1+(1-q)/2)
        return theta_E


class NIE_simple(object):
    """
    this class contains the function and the derivatives of the non-singular isothermal ellipse
    See Keeton&Kochanek 1998
    """
    param_names = ['theta_E', 's', 'q', 'center_x', 'center_y']

    def __init__(self, diff=0.0000000001):
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
        if q >= 1:
            q = 0.999999
        psi = self._psi(x, y, q, s)
        f_x = theta_E / np.sqrt(1. - q ** 2) * np.arctan(np.sqrt(1. - q ** 2) * x / (psi+s))
        f_y = theta_E / np.sqrt(1. - q ** 2) * np.arctanh(np.sqrt(1. - q ** 2) * y / (psi + q**2*s))
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

    @staticmethod
    def _psi(x, y, q, s):
        """
        expression after equation (8) in Keeton&Kochanek 1998

        :param x:
        :param y:
        :param q:
        :param s:
        :return:
        """
        return np.sqrt(q**2 * (s**2 + x**2) + y**2)

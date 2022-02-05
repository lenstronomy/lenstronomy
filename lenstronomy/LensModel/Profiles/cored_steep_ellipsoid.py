__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.Util import util

__all__ = ['CSE']


class CSE(LensProfileBase):
    """
    Cored steep ellipsoid (CSE)
    source:
    Keeton and Kochanek (1998)
    Oguri 2021

    .. math::
        \\kappa(u;s) = \\frac{A}{2(s^2 + \\xi^2)^{3/2}}

    with
    .. math::
        \\xi(x, y) = \\sqrt{x^2 + \\frac{y^2}{q^2}}

    """
    param_names = ['A', 's', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'A': -1000, 's': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'A': 1000, 's': 10000, 'e1': 0.5, 'e2': 0.5, 'center_x': -100, 'center_y': -100}

    def function(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: lensing potential
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        psi = np.sqrt(q**2*(s**2 + x__**2) + y__**2)
        Phi = (psi + s)**2 + (1-q**2) * x__**2
        phi = q/(2*s) * np.log(Phi) - q/s * np.log((1+q) * s)
        return a * phi

    def derivatives(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: deflection in x- and y-direction
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        psi = np.sqrt(q ** 2 * (s ** 2 + x__ ** 2) + y__ ** 2)
        Phi = (psi + s) ** 2 + (1 - q ** 2) * x__ ** 2
        f__x = q * x__ * (psi + q**2*s) / (s * psi * Phi)
        f__y = q * y__ * (psi + s) / (s * psi * Phi)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        return a * f_x, a * f_y

    def hessian(self, x, y, a, s, e1, e2, center_x, center_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param a: lensing strength
        :param s: core radius
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center of profile
        :param center_y: center of profile
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # equations 21-23 in Oguri 2021
        psi = np.sqrt(q ** 2 * (s ** 2 + x__ ** 2) + y__ ** 2)
        Phi = (psi + s) ** 2 + (1 - q ** 2) * x__ ** 2
        f__xx = q/(s * Phi) * (1 + q**2*s*(q**2 * s**2 + y__**2)/psi**3 - 2*x__**2*(psi + q**2*s)**2/(psi**2 * Phi))
        f__yy = q/(s * Phi) * (1 + q**2 * s * (s**2 + x__**2)/psi**3 - 2*y__**2*(psi + s)**2/(psi**2 * Phi))
        f__xy = - q * x__*y__ / (s * Phi) * (q**2 * s / psi**3 + 2 * (psi + q**2*s) * (psi + s) / (psi**2 * Phi))

        # rotate back
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_q) * gamma1__ - np.sin(2 * phi_q) * gamma2__
        gamma2 = +np.sin(2 * phi_q) * gamma1__ + np.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return a * f_xx, a * f_xy, a * f_xy, a * f_yy

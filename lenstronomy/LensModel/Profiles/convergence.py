__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Convergence']


class Convergence(LensProfileBase):
    """
    a single mass sheet (external convergence)
    """
    model_name = 'CONVERGENCE'
    param_names = ['kappa', 'ra_0', 'dec_0']
    lower_limit_default = {'kappa': -10, 'ra_0': -100, 'dec_0': -100}
    upper_limit_default = {'kappa': 10, 'ra_0': 100, 'dec_0': 100}

    def function(self, x, y, kappa, ra_0=0, dec_0=0):
        """
        lensing potential

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa: (external) convergence
        :return: lensing potential
        """
        theta, phi = param_util.cart2polar(x - ra_0, y - dec_0)
        f_ = 1. / 2 * kappa * theta ** 2
        return f_

    def derivatives(self, x, y, kappa, ra_0=0, dec_0=0):
        """
        deflection angle

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa: (external) convergence
        :return: deflection angles (first order derivatives)
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = kappa * x_
        f_y = kappa * y_
        return f_x, f_y

    def hessian(self, x, y, kappa, ra_0=0, dec_0=0):
        """
        Hessian matrix

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa: external convergence
        :param ra_0: zero point of polynomial expansion (no deflection added)
        :param dec_0: zero point of polynomial expansion (no deflection added)
        :return: second order derivatives f_xx, f_xy, f_yx, f_yy
        """
        gamma1 = 0
        gamma2 = 0
        kappa = kappa
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

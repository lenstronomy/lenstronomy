__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
import numpy as np


class MassSheet(object):
    """
    a single mass sheet (external convergence)
    """
    param_names = ['kappa_ext']
    lower_limit_default = {'kappa_ext': -1}
    upper_limit_default = {'kappa_ext': 1}

    def function(self, x, y, kappa_ext):
        """
        lensing potential

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: lensing potential
        """
        theta, phi = param_util.cart2polar(x, y)
        f_ = 1./2 * kappa_ext * theta**2
        return f_

    def derivatives(self, x, y, kappa_ext):
        """
        deflection angle

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: deflection angles (first order derivatives)
        """

        f_x = kappa_ext * x
        f_y = kappa_ext * y
        return f_x, f_y

    def hessian(self, x, y, kappa_ext):
        """
        Hessian matrix

        :param x: x-coordinate
        :param y: y-coordinate
        :param kappa_ext: external convergence
        :return: second order derivatives f_xx, f_yy, f_xy
        """
        gamma1 = 0
        gamma2 = 0
        kappa = kappa_ext
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

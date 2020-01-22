__author__ = 'sibirrer'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import derivative_util as calc_util


class CoredDensity(LensProfileBase):
    """
    class for a uniform cored density dropping steep in the outskirts
    This profile is e.g. featured in Blum et al. 2020 https://arxiv.org/abs/2001.07182v1
    3d rho(r) = 2/pi * Sigma_crit R_c**3 * (R_c**2 + r**2)**(-2)

    """
    param_names = ['sigma0', 'r_core', 'center_x', 'center_y']
    lower_limit_default = {'sigma0': -1, 'r_core': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'sigma0': 10, 'r_core': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, sigma0, r_core, center_x=0, center_y=0):
        """
        potential of cored density profile

        :param x: x-coordinate in angular units
        :param y: y-coordinate in angular units
        :param sigma0: convergence in the core
        :param r_core: core radius
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: lensing potential at (x, y)
        """
        raise ValueError('not implemented')

    def derivatives(self, x, y, sigma0, r_core, center_x=0, center_y=0):
        """
        deflection angle of cored density profile

        :param x: x-coordinate in angular units
        :param y: y-coordinate in angular units
        :param sigma0: convergence in the core
        :param r_core: core radius
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: alpha_x, alpha_y  at (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        alpha_r = self.alpha_r(r, sigma0, r_core)
        f_x = alpha_r * x_ / r
        f_y = alpha_r * y_ / r
        return f_x, f_y

    def hessian(self, x, y, sigma0, r_core, center_x=0, center_y=0):
        """

        :param x: x-coordinate in angular units
        :param y: y-coordinate in angular units
        :param sigma0: convergence in the core
        :param r_core: core radius
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: Hessian df/dxdx, df/dydy, df/dxdy at position (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        d_alpha_dr = self.d_alpha_dr(r, sigma0, r_core)
        alpha = self.alpha_r(r, sigma0, r_core)
        dr_dx = calc_util.d_r_dx(x_, y_)
        dr_dy = calc_util.d_r_dy(x_, y_)
        f_xx = d_alpha_dr * dr_dx * x_ / r + alpha * calc_util.d_x_diffr_dx(x_, y_)
        f_yy = d_alpha_dr * dr_dy * y_ / r + alpha * calc_util.d_y_diffr_dy(x_, y_)
        f_xy = d_alpha_dr * dr_dy * x_ / r + alpha * calc_util.d_x_diffr_dy(x_, y_)
        return f_xx, f_yy, f_xy

    @staticmethod
    def alpha_r(r, sigma0, r_core):
        """
        radial deflection angle of the cored density profile

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: deflection angle
        """
        return 2 * sigma0 * r_core ** 2 / r * (1 - (1 + (r/r_core)**2) **(-1./2))

    def d_alpha_dr(self, r, sigma0, r_core):
        """
        radial derivatives of the radial deflection angle

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: dalpha/dr
        """
        return 2 * sigma0 * (((1 + (r/r_core) ** 2) ** (-3./2)) - (r_core/r) ** 2 * (1 - (1+(r/r_core)**2) ** (-1./2)))

    @staticmethod
    def kappa_r(r, sigma0, r_core):
        """
        convergence of the cored density profile. This routine is also for testing

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: convergence at r
        """
        return sigma0 * (1 + (r/r_core)**2) ** (-3./2)

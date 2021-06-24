__author__ = 'sibirrer'

import numpy as np
from scipy.integrate import quad
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import derivative_util as calc_util

__all__ = ['CoredDensity2']


class CoredDensity2(LensProfileBase):
    """
    class for a uniform cored density dropping steep in the outskirts
    credits for suggesting this profile goes to Kfir Blum

    .. math::
      \\rho(r) = 2/\\pi * \\Sigma_{\\rm crit} R_c^2 * (R_c^2 + r^2)^{-3/2}

    This profile drops like an NFW profile as math:`\\rho(r)^{-3}`.

    """

    model_name = 'CORED_DENSITY_2'
    _s = 0.000001  # numerical limit for minimal radius
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
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        r = np.maximum(r, self._s)
        if isinstance(r, int) or isinstance(r, float):
            return self._num_integral_potential(r, sigma0, r_core)
        else:
            f_ = []
            for i in range(len(r)):
                f_.append(self._num_integral_potential(r[i], sigma0, r_core))
            return np.array(f_)

    def _num_integral_potential(self, r, sigma0, r_core):
        """

        :param r:
        :param r_core:
        :return:
        """
        def _integrand(x):
            return self.alpha_r(x, sigma0=sigma0, r_core=r_core)
        f_ = quad(_integrand, 0, r)[0]
        return f_

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
        r = np.maximum(r, self._s)
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
        :return: Hessian df/dxdx, df/dxdy, df/dydx, df/dydy at position (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        r = np.maximum(r, self._s)
        d_alpha_dr = self.d_alpha_dr(r, sigma0, r_core)
        alpha = self.alpha_r(r, sigma0, r_core)
        dr_dx = calc_util.d_r_dx(x_, y_)
        dr_dy = calc_util.d_r_dy(x_, y_)
        f_xx = d_alpha_dr * dr_dx * x_ / r + alpha * calc_util.d_x_diffr_dx(x_, y_)
        f_yy = d_alpha_dr * dr_dy * y_ / r + alpha * calc_util.d_y_diffr_dy(x_, y_)
        f_xy = d_alpha_dr * dr_dy * x_ / r + alpha * calc_util.d_x_diffr_dy(x_, y_)
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def alpha_r(r, sigma0, r_core):
        """
        radial deflection angle of the cored density profile

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: deflection angle
        """
        return sigma0 * r_core ** 2 * np.log((r_core**2 + r**2) / r_core**2) / r  # this is mass_2d / (r * pi)
        #return sigma0 * r_core * np.arctan(r/r_core)

    @staticmethod
    def d_alpha_dr(r, sigma0, r_core):
        """
        radial derivatives of the radial deflection angle

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: dalpha/dr
        """
        return sigma0 * r_core ** 2 * (-1./r**2 * np.log((r_core**2 + r**2) / r_core**2) + 1/r * r_core**2 / (r**2 + r_core**2) * 2 *r/r_core**2)

    @staticmethod
    def kappa_r(r, sigma0, r_core):
        """
        convergence of the cored density profile. This routine is also for testing

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: convergence at r
        """
        return sigma0 * r_core ** 2 / (r_core ** 2 + r ** 2)

    @staticmethod
    def density(r, sigma0, r_core):
        """
        rho(r) =  2/pi * Sigma_crit R_c**3 * (R_c**2 + r**2)**(-3/2)

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: density at radius r
        """
        return 1./2 * sigma0 * r_core**2 * (r_core**2 + r**2) ** (-3./2)

    def density_lens(self, r, sigma0, r_core):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: density at radius r
        """
        return self.density(r, sigma0, r_core)

    def density_2d(self, x, y, sigma0, r_core, center_x=0, center_y=0):
        """
        projected density at projected radius r

        :param x: x-coordinate in angular units
        :param y: y-coordinate in angular units
        :param sigma0: convergence in the core
        :param r_core: core radius
        :param center_x: center of the profile
        :param center_y: center of the profile
        :return: projected density
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        r = np.maximum(r, self._s)
        return self.kappa_r(r, sigma0, r_core)

    def mass_2d(self, r, sigma0, r_core):
        """
        mass enclosed in cylinder of radius r

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: mass enclosed in cylinder of radius r
        """
        return sigma0 * r_core ** 2 * np.pi * np.log((r_core**2 + r**2) / r_core**2)

    @staticmethod
    def mass_3d(r, sigma0, r_core):
        """
        mass enclosed 3d radius

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: mass enclosed 3d radius
        """
        r_ = np.sqrt(r**2 + r_core**2)
        return 2 * np.pi * sigma0 * r_core**2 * (r_ * np.log(r_ + r) - np.log(r_core) * r_ - r) / r_

    def mass_3d_lens(self, r, sigma0, r_core):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units
        For this profile those are identical.

        :param r: radius (angular scale)
        :param sigma0: convergence in the core
        :param r_core: core radius
        :return: mass enclosed 3d radius
        """
        return self.mass_3d(r, sigma0, r_core)

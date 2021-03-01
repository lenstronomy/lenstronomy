__author__ = 'dangilman'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from scipy.special import hyp2f1
from scipy.special import gamma as gamma_func

__all__ = ['SPLCORE']


class SPLCORE(LensProfileBase):
    """
    This lens profile corresponds to a spherical power law (CPL) mass distribution with logarithmic slope gamma (g) and
    a 3D core radius rc

    rho(r, rc, g) = rho0 * (rc ^ g / (rc^2 + r^2)^(g/2))

    The difference between this and EPL is that this model contains a core radius, is circular, and is defined for gamma=3.

    With respect to SPEMD, this model is different in that it is defined for gamma = 3, is circular, and written in terms
    of physical density parameter rho0, or the central density at r=0

    This class is defined for gamma > 1
    """

    param_names = ['rho0', 'center_x', 'center_y', 'rc', 'gamma']
    lower_limit_default = {'rho0': 0, 'center_x': -100, 'center_y': -100, 'rc': 1e-6, 'gamma': 1.+1e-6}
    upper_limit_default = {'rho0': 1e+12, 'center_x': 100, 'center_y': 100, 'rc': 100, 'gamma': 5.}

    def function(self, x, y, sigma0, rc, gamma, center_x=0, center_y=0):

        raise Exception('potential not implemented for this class')

    def derivatives(self, x, y, sigma0, rc, gamma, center_x=0, center_y=0):
        """
        :param x: projected x position at which to evaluate function [arcsec]
        :param y: projected y position at which to evaluate function [arcsec]
        :param sigma0: convergence at r = 0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :param center_x: x coordinate center of lens model [arcsec]
        :param center_y: y coordinate center of lens model [arcsec]
        :return: deflection angle alpha in x and y directions

        alpha_(x/y) = alpha_r * cos/sin(x/y / r)
        """

        x_ = x - center_x
        y_ = y - center_y
        r = np.hypot(x_, y_)
        alpha_r = self.alpha(r, sigma0, rc, gamma)
        cos = x_/r
        sin = y_/r
        return alpha_r * cos, alpha_r * sin

    def hessian(self, x, y, sigma0, rc, gamma, center_x=0, center_y=0):
        """
        :param x: projected x position at which to evaluate function [arcsec]
        :param y: projected y position at which to evaluate function [arcsec]
        :param sigma0: convergence at r = 0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :param center_x: x coordinate center of lens model [arcsec]
        :param center_y: y coordinate center of lens model [arcsec]
        :return: deflection angle alpha in x and y directions

        alpha_(x/y) = alpha_r * cos/sin(x/y / r)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_ ** 2 + y_ ** 2)
        m2d = self.mass_2d_lens(r, sigma0, rc, gamma) / np.pi

        rho0 = self._sigma2rho0(sigma0, rc)
        kappa = self.density_2d(x_, y_, rho0, rc, gamma)
        gamma_tot = m2d / r ** 2 - kappa
        sin_2phi = -2*x_*y_/r**2
        cos_2phi = (y_**2 - x_**2)/r**2

        gamma1 = cos_2phi * gamma_tot
        gamma2 = sin_2phi * gamma_tot

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def alpha(self, r, sigma0, rc, gamma):

        """
        Returns the deflection angle at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: deflection angle at r
        """
        mass2d = self.mass_2d_lens(r, sigma0, rc, gamma)
        return mass2d / r / np.pi


    def density(self, r, rho0, rc, gamma):
        """
        Returns the 3D density at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: density at r
        """
        return rho0 * rc ** gamma / (rc**2 + r**2) ** (gamma/2)

    def density_lens(self, r, sigma0, rc, gamma):
        """
        Returns the 3D density at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: density at r
        """
        rho0 = self._sigma2rho0(sigma0, rc)
        return rho0 * rc ** gamma / (rc**2 + r**2) ** (gamma/2)

    def density_2d(self, x, y, rho0, rc, gamma):
        """
        Returns the convergence at radius r
        :param x: x position [arcsec]
        :param y: y position [arcsec]
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: convergence at r
        """

        r = np.hypot(x, y)
        sigma0 = self._rho02sigma(rho0, rc)
        if gamma == 3:
            return 2 * rc ** 2 * sigma0 / (r ** 2 + rc ** 2)
        elif gamma == 2:
            return np.pi * rc * sigma0 / (rc**2 + r**2) ** 0.5
        else:
            x = r / rc
            exponent = (1 - gamma) / 2
            return sigma0 * np.sqrt(np.pi) * gamma_func(0.5 * (gamma - 1)) / gamma_func(0.5 * gamma) * (1 + x ** 2) ** exponent

    def mass_3d(self,r, rho0, rc, gamma):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius [arcsec]
        :param rho0: density at r = 0 in units [rho_0_physical / sigma_crit] (which should be equal to [arcsec])
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: mass inside radius r
        """
        return 4 * np.pi * rc ** 3 * rho0 * self._g(r/rc, gamma)

    def mass_3d_lens(self, r, sigma0, rc, gamma):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius [arcsec]
        :param sigma0: convergence at r = 0
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: mass inside radius r
        """
        rho0 = self._sigma2rho0(sigma0, rc)
        return self.mass_3d(r, rho0, rc, gamma)

    def mass_2d(self, r, rho0, rc, gamma):
        """
        mass enclosed projected 2d disk of radius r
        :param r: radius [arcsec]
        :param rho0: density at r = 0 in units [rho_0_physical / sigma_crit] (which should be equal to [arcsec^{-1}])
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: projected mass inside disk of radius r
        """
        return 4 * np.pi * rc ** 3 * rho0 * self._f(r/rc, gamma)

    def mass_2d_lens(self, r, sigma0, rc, gamma):
        """
        mass enclosed projected 2d disk of radius r
        :param r: radius [arcsec]
        :param sigma0: convergence at r = 0
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param rc: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: projected mass inside disk of radius r
        """
        rho0 = self._sigma2rho0(sigma0, rc)
        return self.mass_2d(r, rho0, rc, gamma)

    @staticmethod
    def _sigma2rho0(sigma0, rc):
        """
        Converts the convergence normalization to the 3d normalization
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :return: density normalization in units arcsec^-1, or rho_0_physical / sigma_crit
        """
        return sigma0 / rc

    @staticmethod
    def _rho02sigma(rho0, rc):
        """
        Converts the convergence normalization to the 3d normalization
        :param sigma0: convergence at r=0
        :param rc: core radius [arcsec]
        :return: density normalization in units arcsec^-1, or rho_0_physical / sigma_crit
        """
        return rho0 * rc

    @staticmethod
    def _f(x, gamma):
        """
        Returns the solution of the 2D mass integral
        :param x: dimensionaless radius r/rc
        :param gamma: logarithmic slope at r -> infinity
        :return: a number
        """
        if gamma == 3:
            return 0.5 * np.log(x ** 2 + 1)
        elif gamma == 2:
            return np.pi/2 * ((x**2 + 1)**0.5 - 1)
        else:
            gamma_func_term = gamma_func(0.5 * (gamma - 1)) / gamma_func(0.5 * gamma)
            prefactor = np.sqrt(np.pi) * gamma_func_term / (2 * (gamma - 3))
            term = (1 - (1 + x ** 2) ** ((3 - gamma) / 2))
            return prefactor * term

    @staticmethod
    def _g(x, gamma):
        """
        Returns the solution of the 3D mass integral
        :param x: dimensionaless radius r/rc
        :param gamma: logarithmic slope at r -> infinity
        :return: a number
        """

        if gamma == 3:
            return np.arcsinh(x) - x / (1 + x ** 2) ** 0.5
        elif gamma == 2:
            return x - np.arctan(x)
        else:
            prefactor = 1 / ((gamma - 3) * (gamma - 1)) / x
            term = hyp2f1(-0.5, gamma / 2, 0.5, -x ** 2) - (1 + x ** 2) ** ((2 - gamma) / 2) * (
                        1 + x ** 2 * (gamma - 1))
            return prefactor * term

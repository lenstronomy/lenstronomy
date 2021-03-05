__author__ = 'dangilman'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from scipy.special import hyp2f1
from scipy.special import gamma as gamma_func

__all__ = ['SPLCORE']


class SPLCORE(LensProfileBase):
    """
    This lens profile corresponds to a spherical power law (SPL) mass distribution with logarithmic slope gamma and
    a 3D core radius r_core

    .. math::

        \rho\left(r, \rho_0, r_c, \gamma\right) = \rho_0  \frac{{r_c}^\gamma}{\left(r^2 + r_c^2\right)^{\frac{\gamma}{2}}}

    The difference between this and EPL is that this model contains a core radius, is circular, and is also defined for gamma=3.

    With respect to SPEMD, this model is different in that it is also defined for gamma = 3, is circular, and is defined
    in terms of a physical density parameter rho0, or the central density at r=0 divided by the critical density for lensing
    such that rho0 has units 1/arcsec.

    This class is defined for all gamma > 1
    """

    param_names = ['sigma0', 'center_x', 'center_y', 'r_core', 'gamma']
    lower_limit_default = {'sigma0': 0, 'center_x': -100, 'center_y': -100, 'r_core': 1e-6, 'gamma': 1.+1e-6}
    upper_limit_default = {'sigma0': 1e+12, 'center_x': 100, 'center_y': 100, 'r_core': 100, 'gamma': 5.}

    def function(self, x, y, sigma0, r_core, gamma, center_x=0, center_y=0):

        raise Exception('potential not implemented for this class')

    def derivatives(self, x, y, sigma0, r_core, gamma, center_x=0, center_y=0):
        """
        :param x: projected x position at which to evaluate function [arcsec]
        :param y: projected y position at which to evaluate function [arcsec]
        :param sigma0: convergence at r = 0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :param center_x: x coordinate center of lens model [arcsec]
        :param center_y: y coordinate center of lens model [arcsec]
        :return: deflection angle alpha in x and y directions
        """

        x_ = x - center_x
        y_ = y - center_y
        r = np.hypot(x_, y_)
        r = self._safe_r_division(r, r_core)

        alpha_r = self.alpha(r, sigma0, r_core, gamma)
        cos = x_/r
        sin = y_/r
        return alpha_r * cos, alpha_r * sin

    def hessian(self, x, y, sigma0, r_core, gamma, center_x=0, center_y=0):
        """
        :param x: projected x position at which to evaluate function [arcsec]
        :param y: projected y position at which to evaluate function [arcsec]
        :param sigma0: convergence at r = 0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :param center_x: x coordinate center of lens model [arcsec]
        :param center_y: y coordinate center of lens model [arcsec]
        :return: hessian elements

        alpha_(x/y) = alpha_r * cos/sin(x/y / r)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.hypot(x_, y_)
        m2d = self.mass_2d_lens(r, sigma0, r_core, gamma) / np.pi
        r = self._safe_r_division(r, r_core)

        rho0 = self._sigma2rho0(sigma0, r_core)
        kappa = self.density_2d(x_, y_, rho0, r_core, gamma)
        gamma_tot = m2d / r ** 2 - kappa
        sin_2phi = -2*x_*y_/r**2
        cos_2phi = (y_**2 - x_**2)/r**2

        gamma1 = cos_2phi * gamma_tot
        gamma2 = sin_2phi * gamma_tot

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    def alpha(self, r, sigma0, r_core, gamma):

        """
        Returns the deflection angle at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: deflection angle at r
        """
        mass2d = self.mass_2d_lens(r, sigma0, r_core, gamma)
        return mass2d / r / np.pi

    def density(self, r, rho0, r_core, gamma):
        """
        Returns the 3D density at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: density at r
        """
        return rho0 * r_core ** gamma / (r_core**2 + r**2) ** (gamma/2)

    def density_lens(self, r, sigma0, r_core, gamma):
        """
        Returns the 3D density at r
        :param r: radius [arcsec]
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: density at r
        """
        rho0 = self._sigma2rho0(sigma0, r_core)
        return rho0 * r_core ** gamma / (r_core**2 + r**2) ** (gamma/2)

    def _density_2d_r(self, r, rho0, r_core, gamma):
        """
        Returns the convergence at radius r after applying _safe_r_division
        :param r: position [arcsec]
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: convergence at r
        """
        sigma0 = self._rho02sigma(rho0, r_core)
        if gamma == 3:
            return 2 * r_core ** 2 * sigma0 / (r ** 2 + r_core ** 2)
        elif gamma == 2:
            return np.pi * r_core * sigma0 / (r_core ** 2 + r ** 2) ** 0.5
        else:
            x = r / r_core
            exponent = (1 - gamma) / 2
            return sigma0 * np.sqrt(np.pi) * gamma_func(0.5 * (gamma - 1)) / gamma_func(0.5 * gamma) * (
                        1 + x ** 2) ** exponent

    def density_2d(self, x, y, rho0, r_core, gamma):
        """
        Returns the convergence at radius r
        :param x: x position [arcsec]
        :param y: y position [arcsec]
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: convergence at r
        """

        r = np.hypot(x, y)
        r = self._safe_r_division(r, r_core)
        return self._density_2d_r(r, rho0, r_core, gamma)

    def mass_3d(self, r, rho0, r_core, gamma):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius [arcsec]
        :param rho0: density at r = 0 in units [rho_0_physical / sigma_crit] (which should be equal to [arcsec])
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: mass inside radius r
        """
        return 4 * np.pi * r_core ** 3 * rho0 * self._g(r/r_core, gamma)

    def mass_3d_lens(self, r, sigma0, r_core, gamma):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius [arcsec]
        :param sigma0: convergence at r = 0
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: mass inside radius r
        """
        rho0 = self._sigma2rho0(sigma0, r_core)
        return self.mass_3d(r, rho0, r_core, gamma)

    def mass_2d(self, r, rho0, r_core, gamma):
        """
        mass enclosed projected 2d disk of radius r
        :param r: radius [arcsec]
        :param rho0: density at r = 0 in units [rho_0_physical / sigma_crit] (which should be equal to [1/arcsec])
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: projected mass inside disk of radius r
        """
        return 4 * np.pi * r_core ** 3 * rho0 * self._f(r/r_core, gamma)

    def mass_2d_lens(self, r, sigma0, r_core, gamma):
        """
        mass enclosed projected 2d disk of radius r
        :param r: radius [arcsec]
        :param sigma0: convergence at r = 0
        where rho_0_physical is a physical density normalization and sigma_crit is the critical density for lensing
        :param r_core: core radius [arcsec]
        :param gamma: logarithmic slope at r -> infinity
        :return: projected mass inside disk of radius r
        """
        rho0 = self._sigma2rho0(sigma0, r_core)
        return self.mass_2d(r, rho0, r_core, gamma)

    @staticmethod
    def _safe_r_division(r, r_core, x_min=1e-6):
        """
        Avoids accidental division by 0
        :param r: radius in arcsec
        :param r_core: core radius in arcsec
        :return: a minimum value of r
        """

        if isinstance(r, float) or isinstance(r, int):
            r = max(x_min * r_core, r)
        else:
            r[np.where(r < x_min * r_core)] = x_min * r_core
        return r

    @staticmethod
    def _sigma2rho0(sigma0, r_core):
        """
        Converts the convergence normalization to the 3d normalization
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :return: density normalization in units 1/arcsec, or rho_0_physical / sigma_crit
        """
        return sigma0 / r_core

    @staticmethod
    def _rho02sigma(rho0, r_core):
        """
        Converts the convergence normalization to the 3d normalization
        :param sigma0: convergence at r=0
        :param r_core: core radius [arcsec]
        :return: density normalization in units 1/arcsec, or rho_0_physical / sigma_crit
        """
        return rho0 * r_core

    @staticmethod
    def _f(x, gamma):
        """
        Returns the solution of the 2D mass integral defined such that

        .. math::

            m_{\rm{2D}}\left(R\right) = 4 \pi r_{\rm{core}}^3 \rho_0 F\left(\frac{R}{r_{\rm{core}}}, \gamma\right)

        :param x: dimensionaless radius r/r_core
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
        Returns the solution of the 3D mass integral defined such that
        Returns the solution of the 2D mass integral defined such that

        .. math::
            m_{\rm{3D}}\left(R\right) = 4 \pi r_{\rm{core}}^3 \rho_0 G\left(\frac{R}{r_{\rm{core}}}, \gamma\right)

        :param x: dimensionaless radius r/r_core
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

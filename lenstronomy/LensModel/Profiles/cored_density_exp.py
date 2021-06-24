__author__ = 'lucateo'

import numpy as np
import scipy.interpolate as interp
from scipy.special import exp1, erf
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const

__all__ = ['CoredDensityExp']


class CoredDensityExp(LensProfileBase):
    """
    this class contains functions concerning an exponential cored density profile,
    namely

    ..math::
        \\rho(r) = \\rho_0 \\exp(- (\\theta / \\theta_c)^2)

    """
    _s = 0.000001  # numerical limit for minimal radius
    param_names = ['kappa_0', 'theta_c', 'center_x', 'center_y']
    lower_limit_default = {'kappa_0': 0, 'theta_c': 0, 'center_x': -100, 'center_y': -100 }
    upper_limit_default = {'kappa_0': 10, 'theta_c': 100, 'center_x': 100, 'center_y': 100 }

    def rhotilde(self, kappa_0, theta_c):
        """
        Computes the central density in angular units
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: central density in 1/arcsec
        """
        return kappa_0 / (np.sqrt(np.pi) * theta_c)

    def function(self, x, y, kappa_0, theta_c, center_x=0, center_y=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential (in arcsec^2)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_** 2 + y_** 2)
        r = np.maximum(r, self._s)
        Integral_factor = 0.5 * exp1( (r/theta_c)**2) + np.log( (r/theta_c))
        function = kappa_0 * theta_c**2 * Integral_factor
        return function

    @staticmethod
    def alpha_radial(r, kappa_0, theta_c):
        """
        returns the radial part of the deflection angle
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: radial deflection angle
        """
        prefactor = kappa_0 * theta_c**2 / r
        return prefactor * (1 - np.exp(- (r/theta_c)**2 ))

    def derivatives(self, x, y, kappa_0, theta_c, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (lensing potential), which are
        the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        f_x = self.alpha_radial(R, kappa_0, theta_c ) * x_ / R
        f_y = self.alpha_radial(R, kappa_0, theta_c) * y_ / R
        return f_x, f_y

    def hessian(self, x, y, kappa_0, theta_c, center_x=0, center_y=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        prefactor = kappa_0 * theta_c**2
        expFactor = np.exp( - (R/theta_c)**2)
        factor1 = (1 - expFactor)/R**4
        factor2 = 2/(R**2 * theta_c**2) * expFactor
        f_xx = prefactor * ( factor1 * (y_**2 - x_**2) + factor2 * x_**2 )
        f_yy = prefactor * ( factor1 * (x_**2 - y_**2) + factor2 * y_**2 )
        f_xy = prefactor * ( - factor1 * 2 * x_ * y_ + factor2 *x_*y_ )
        return f_xx, f_xy, f_xy, f_yy

    def density(self, R, kappa_0, theta_c):
        """
        three dimensional density profile in angular units
        (rho0_physical = rho0_angular Sigma_crit / D_lens)

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: rho(R) density
        """
        rhotilde = self.rhotilde(kappa_0, theta_c)
        return rhotilde * np.exp( -(R/theta_c)**2)

    def density_lens(self, r, kappa_0, theta_c):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: density rho(r)
        """
        return self.density(r, kappa_0, theta_c)

    def kappa_r(self, R, kappa_0, theta_c):
        """
        convergence of the cored density profile. This routine is also for testing

        :param R: radius (angular scale)
        :param kappa_0: convergence in the core
        :param theta_c: core radius
        :return: convergence at r
        """
        expFactor = np.exp( - (R/theta_c)**2)
        return kappa_0  * expFactor

    def density_2d(self, x, y, kappa_0, theta_c, center_x = 0, center_y = 0):
        """
        projected two dimensional ULDM profile (convergence * Sigma_crit), but given our
        units convention for rho0, it is basically the convergence

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        return self.kappa_r(R, kappa_0, theta_c)

    @staticmethod
    def mass_3d( R, kappa_0, theta_c):
        """
        mass enclosed a 3d sphere or radius r
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param R: radius in arcseconds
        :return: mass of soliton in angular units
        """
        integral_factor = np.sqrt(np.pi) * erf(R/theta_c)/2 - R/theta_c * np.exp(-(R/theta_c)**2)
        m_3d =  2* np.sqrt(np.pi) * kappa_0 * theta_c**2 * integral_factor
        return m_3d

    def mass_3d_lens(self, r, kappa_0, theta_c):
        """
        mass enclosed a 3d sphere or radius r
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: mass
        """
        m_3d = self.mass_3d(r, kappa_0, theta_c)
        return m_3d

    def mass_2d(self, R, kappa_0, theta_c):
        """
        mass enclosed a 2d sphere of radius r
        returns

        .. math::
            M_{2D} = 2 \\pi \\int_0^r dr' r' \\int dz \\rho(\\sqrt(r'^2 + z^2))

        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: M_2D (ULDM only)
        """
        return self.alpha_radial(R, kappa_0, theta_c) * np.pi * R


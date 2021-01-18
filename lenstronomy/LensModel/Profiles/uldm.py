__author__ = 'lucateo'

# this file contains a class to compute the Ultra Light Dark Matter soliton profile
import numpy as np
import scipy.interpolate as interp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const
__all__ = ['ULDM']


class Uldm(LensProfileBase):
    """
    This class contains functions concerning the ULDM soliton density profile, whose good approximation is
    \rho = \rho_0 (1 + 0.091(r/r_core)^2)^-8
    it has, as parameters:
    :param kappa_0: central convergence
    :param theta_c: core radius (in arcseconds)
    """
    _s = 0.000001  # numerical limit for minimal radius
    param_names = ['kappa_0', 'theta_c', 'center_x', 'center_y']
    # rule of thumb: m_phys = 10^-15 m_noCosmo, M_phys = 10^21 M_noCosmo
    lower_limit_default = {'kappa_0': 0, 'theta_c': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'kappa_0': 1., 'theta_c': 100, 'center_x': 100, 'center_y': 100}

    def rhoTilde(self, kappa_0, theta_c):
        """
        Computes the central density in angular units
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: central density in 1/arcsec
        """
        num_factor = 2048 * np.sqrt(0.091) /(429 * np.pi)
        return kappa_0 * num_factor / theta_c

    def lensing_Integral(self, x):
        """
        The analitic result of the integral entering the computation of the lensing potential, that is
        \int dy/y (1 - (1 + y^2)^(-13/2))
        :param x: evaluation point of the integral
        :return: result of the antiderivative in x
        """
        denominator = 3465*(x**2 +1)**(5.5)
        numerator = 3465*x**10 + 18480*x**8 + 39963*x**6 + 44154*x**4 + 25399*x**2 + 6508
        return np.log(np.sqrt(x**2 +1) + 1) - numerator/denominator

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
        Integral_factor = self.lensing_Integral(np.sqrt(0.091) * r / theta_c)
        prefactor = 2/ 13 * kappa_0 * theta_c**2 / 0.091
        return prefactor * Integral_factor

    def alpha_radial(self, r, kappa_0, theta_c):
        """
        returns the radial part of the deflection angle

        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param r: radius where the deflection angle is computed
        :return: radial deflection angle
        """
        prefactor = 2/ 13 * kappa_0 * theta_c**2 / 0.091
        denominator_factor = (1 + 0.091 * r**2/theta_c**2)**(6.5)
        return prefactor/r * (1 - 1/denominator_factor)

    def derivatives(self, x, y, kappa_0, theta_c, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (lensing potential), which are the deflection angles

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
        f_x = self.alpha_radial(R, kappa_0, theta_c) * x_ / R
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
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        prefactor = 2/ 13 * kappa_0 * theta_c**2 / 0.091
        # denominator factor
        denominator = 1 + 0.091 * R**2/theta_c**2
        factor1 = 13 * 0.091 * denominator**(-7.5) / (theta_c**2 * R**2)
        factor2 = 1/R**4 * (1 - denominator**(-6.5))
        f_xx = prefactor * (factor1 * x_**2 + factor2 * (y_**2 - x_**2))
        f_yy = prefactor * (factor1 * y_**2 + factor2 * (x_**2 - y_**2))
        f_xy = prefactor * (factor1 * x_ * y_ - factor2 * 2*x_*y_)
        return f_xx, f_yy, f_xy

    def density(self, R, kappa_0, theta_c):
        """
        three dimensional ULDM profile in angular units (rho0_physical = rho0_angular \Sigma_crit / D_lens)
        :param R: radius of interest
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: rho(R) density in angular units
        """
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        return rhotilde/(1 + 0.091* (R/theta_c)**2)**8

    def density_lens(self, r, kappa_0, theta_c):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: 3d radius
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
        return kappa_0  * (1 + 0.091 * (R/theta_c)**2)**(-15/2)


    def density_2d(self, x, y, kappa_0, theta_c, center_x=0, center_y=0):
        """
        projected two dimensional ULDM profile (convergence * \Sigma_crit), but given our
        units convention for rho0, it is basically the convergence

        :param R: radius of interest
        :type R: float/numpy array
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        return self.kappa_r(R, kappa_0, theta_c)

    def mass_Integral(self, x):
        """
        Returns the analitic result of the integral appearing in mass expression
        """
        numerator = x * (3465 * x**12 + 23100 * x**10 + 65373 * x**8 + 101376*x**6 + 92323*x**4 + 48580 * x**2 - 3465)
        denominator = 215040 * (x**2 + 1)**7
        result = 33 * np.arctan(x) / 2048 + numerator/denominator
        return result

    def mass_3d(self, R, kappa_0, theta_c):
        """
        mass enclosed a 3d sphere or radius r
        :param R: radius in arcseconds
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: mass of soliton in angular units
        """
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        m_3d = 4. * np.pi * rhotilde * theta_c**3 / (0.091)**(1.5) * (self.mass_Integral(R/theta_c * np.sqrt(0.091)) - self.mass_Integral(0) )
        return m_3d

    def mass_3d_lens(self, r, kappa_0, theta_c):
        """
        mass enclosed a 3d sphere or radius r
        :param R:
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param rho0: central density in angular units
        :return: mass
        """
        m_3d = self.mass_3d(r, kappa_0, theta_c)
        return m_3d

    def mass_2d(self, R, kappa_0, theta_c):
        """
        mass enclosed a 2d sphere or radius r
        :param R: radius over which the mass is computed
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :return: mass enclosed in 2d sphere
        """
        m_2d = np.pi * R * self.alpha_radial(R, kappa_0, theta_c)
        return m_2d

__author__ = 'lucateo'

# this file contains a class to compute the Ultra Light Dark Matter soliton profile
import numpy as np
import scipy.interpolate as interp
from scipy.special import gamma, hyp2f1
from mpmath import hyp3f2
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const
__all__ = ['Uldm']


class Uldm(LensProfileBase):
    """
    This class contains functions concerning the ULDM soliton density profile,
    whose good approximation is (see for example https://arxiv.org/pdf/1406.6586.pdf )

    .. math::

        \\rho = \\rho_0 (1 + a(\\theta/\\theta_c)^2)^{-\\beta}

    where :math:`\\theta_c` is the core radius, corresponding to the radius where the
    density drops by half its central value, :math: `\\beta` is the slope (called just slope 
    in the parameters of this model), :math: `\\rho_0 = \\kappa_0 \\Sigma_c/D_lens`,
    and :math: `a` is a parameter, dependent on :math: `\\beta`, chosen such
    that :math: `\\theta_c` indeed corresponds to the radius where the density drops by half
    (simple math gives :math: `a = 0.5^{-1/\\beta} - 1` ).
    For an ULDM soliton profile without contributions to background potential, it
    turns out that :math: `\\beta = 8, a = 0.091`. We allow :math: `\\beta` to be 
    different from 8 to model solitons which feel the influence of background 
    potential (see 2105.10873)
    The profile has, as parameters:
    :param kappa_0: central convergence
    :param theta_c: core radius (in arcseconds)
    :param slope: exponent entering the profile, default value is 8
    """
    _s = 0.000001  # numerical limit for minimal radius
    param_names = ['kappa_0', 'theta_c', 'slope', 'center_x', 'center_y']
    lower_limit_default = {'kappa_0': 0, 'theta_c': 0, 'slope': 3.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'kappa_0': 1., 'theta_c': 100, 'slope': 10, 'center_x': 100, 'center_y': 100}

    def rhotilde(self, kappa_0, theta_c, slope=8):
        """
        Computes the central density in angular units
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: central density in 1/arcsec
        """
        a_factor_sqrt = np.sqrt( (0.5)**(-1/slope) -1)
        num_factor = gamma(slope) / gamma(slope - 1/2) * a_factor_sqrt / np.sqrt(np.pi)
        return kappa_0 * num_factor / theta_c

    def function(self, x, y, kappa_0, theta_c, center_x=0, center_y=0, slope=8):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential (in arcsec^2)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_** 2 + y_** 2)
        r = np.maximum(r, self._s)
        a_factor_sqrt = np.sqrt( (0.5)**(-1./slope) -1)
        if np.isscalar(r) == True:
            hypgeom = float(kappa_0 /2 * r**2 * 
                hyp3f2(1, 1, slope - 0.5, 2, 2, -(a_factor_sqrt * r /theta_c )**2))
        else:
            hypgeom =  np.array([ kappa_0 /2. * r_i**2. *
                hyp3f2(1, 1, slope - 0.5, 2, 2, -(a_factor_sqrt * r_i / theta_c)**2.) for r_i in r], dtype=float)
        return hypgeom

    def alpha_radial(self, r, kappa_0, theta_c, slope=8):
        """
        returns the radial part of the deflection angle

        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :param r: radius where the deflection angle is computed
        :return: radial deflection angle
        """
        a_factor =  (0.5)**(-1./slope) -1
        prefactor = 2./(2*slope -3) * kappa_0 * theta_c**2 / a_factor
        denominator_factor = (1 + a_factor * r**2/theta_c**2)**(slope - 3./2)
        return prefactor/r * (1 - 1/denominator_factor)

    def derivatives(self, x, y, kappa_0, theta_c, center_x=0, center_y=0, slope=8):
        """
        returns df/dx and df/dy of the function (lensing potential), which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        f_x = self.alpha_radial(R, kappa_0, theta_c, slope) * x_ / R
        f_y = self.alpha_radial(R, kappa_0, theta_c, slope) * y_ / R
        return f_x, f_y

    def hessian(self, x, y, kappa_0, theta_c, center_x=0, center_y=0, slope=8):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        a_factor =  (0.5)**(-1./slope) -1
        prefactor = 2./(2*slope -3) * kappa_0 * theta_c**2 / a_factor
        # denominator factor
        denominator = 1 + a_factor * R**2/theta_c**2
        factor1 = (2*slope - 3) * a_factor * denominator**(1./2 - slope) / (theta_c**2 * R**2)
        factor2 = 1/R**4 * (1 - denominator**(3./2 - slope))
        f_xx = prefactor * (factor1 * x_**2 + factor2 * (y_**2 - x_**2))
        f_yy = prefactor * (factor1 * y_**2 + factor2 * (x_**2 - y_**2))
        f_xy = prefactor * (factor1 * x_ * y_ - factor2 * 2*x_*y_)
        return f_xx, f_xy, f_xy, f_yy

    def density(self, R, kappa_0, theta_c, slope=8):
        """
        three dimensional ULDM profile in angular units
        (rho0_physical = rho0_angular Sigma_crit / D_lens)
        :param R: radius of interest
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: rho(R) density in angular units
        """
        rhotilde = self.rhotilde(kappa_0, theta_c, slope)
        a_factor =  (0.5)**(-1./slope) -1
        return rhotilde/(1 + a_factor* (R/theta_c)**2)**slope

    def density_lens(self, r, kappa_0, theta_c, slope=8):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the
        convergence quantity.

        :param r: 3d radius
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: density rho(r)
        """
        return self.density(r, kappa_0, theta_c, slope)

    def kappa_r(self, R, kappa_0, theta_c, slope=8):
        """
        convergence of the cored density profile. This routine is also for testing

        :param R: radius (angular scale)
        :param kappa_0: convergence in the core
        :param theta_c: core radius
        :param slope: exponent entering the profile
        :return: convergence at r
        """
        a_factor =  (0.5)**(-1./slope) -1
        return kappa_0  * (1 + a_factor * (R/theta_c)**2)**(1./2 - slope)


    def density_2d(self, x, y, kappa_0, theta_c, center_x=0, center_y=0, slope=8):
        """
        projected two dimensional ULDM profile (convergence * Sigma_crit), but
        given our units convention for rho0, it is basically the convergence

        :param R: radius of interest
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        return self.kappa_r(R, kappa_0, theta_c, slope)

    def _mass_integral(self, x, slope=8):
        """
        Returns the analitic result of the integral appearing in mass expression
        :param slope: exponent entering the profile
        :return: integral result
        """
        hypF = np.real(hyp2f1(3./2, slope, 5./2, - x**2))
        return 1./3 * x**3 * hypF

    def mass_3d(self, R, kappa_0, theta_c, slope=8):
        """
        mass enclosed a 3d sphere or radius r
        :param R: radius in arcseconds
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: mass of soliton in angular units
        """
        rhotilde = self.rhotilde(kappa_0, theta_c, slope)
        a_factor =  (0.5)**(-1./slope) -1
        prefactor = 4. * np.pi * rhotilde * theta_c**3 / (a_factor)**(1.5)
        m_3d = prefactor * (self._mass_integral(R/theta_c * np.sqrt(a_factor), slope)
                - self._mass_integral(0, slope) )
        return m_3d

    def mass_3d_lens(self, r, kappa_0, theta_c, slope=8):
        """
        mass enclosed a 3d sphere or radius r
        :param r: radius over which the mass is computed
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: mass enclosed in 3D ball
        """
        m_3d = self.mass_3d(r, kappa_0, theta_c, slope)
        return m_3d

    def mass_2d(self, R, kappa_0, theta_c, slope=8):
        """
        mass enclosed a 2d sphere or radius r
        :param R: radius over which the mass is computed
        :param kappa_0: central convergence of profile
        :param theta_c: core radius (in arcsec)
        :param slope: exponent entering the profile
        :return: mass enclosed in 2d sphere
        """
        m_2d = np.pi * R * self.alpha_radial(R, kappa_0, theta_c, slope)
        return m_2d

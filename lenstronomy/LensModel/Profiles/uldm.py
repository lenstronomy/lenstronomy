__author__ = 'lucateo'

# this file contains a class to compute the Ultra Light Dark Matter soliton profile
import numpy as np
import scipy.interpolate as interp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const
__all__ = ['ULDM']


class Uldm(LensProfileBase):
    """
    This class contains functions concerning the ULDM soliton density profile; it
    has, as parameters:
        :param m_noCosmo_log10: it is \log_10 (m \sqrt{\Sigma_crit D_lens^3} ), m in eV, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :param M_noCosmo_log10: it is \log_10 ( M/(D_lens^2 \Sigma_crit) ), M in M_sun, \Sigma_crit in M_sun / pc^2, D_lens in pc
    this seemingly weird definition are needed in order not to make cosmology enter
    in this class (hence keeping everything in angular units).
    """
    _s = 0.000001  # numerical limit for minimal radius
    param_names = ['m_noCosmo_log10', 'M_noCosmo_log10', 'center_x', 'center_y']
    # rule of thumb: m_phys = 10^-15 m_noCosmo, M_phys = 10^21 M_noCosmo
    lower_limit_default = {'m_noCosmo_log10': -12, 'M_noCosmo_log10': -15, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'m_noCosmo_log10': -3, 'M_noCosmo_log10': -6, 'center_x': 100, 'center_y': 100}

    def theta_cRad(self, m_noCosmo_log10, M_noCosmo_log10):
        """
        theta core in radiant
        :param m_noCosmo_log10: it is \log_10 (m \sqrt{\Sigma_crit D_lens^3} ), m in eV, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :param M_noCosmo_log10: it is \log_10 ( M/(D_lens^2 \Sigma_crit) ), M in M_sun, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :return: \theta_c in radiants
        """
        m22 = 10**(m_noCosmo_log10 +22)
        M9 = 10**(M_noCosmo_log10 -9)
        return 160 * 1.4 * M9**(-1) * m22**(-2)

    def rho0Tilde(self, m_noCosmo_log10, M_noCosmo_log10):
        """
        rho0 tilde = rho0 * D_lens / \Sigma_crit, central density in angular units
        :param m_noCosmo_log10: it is \log_10 (m \sqrt{\Sigma_crit D_lens^3} ), m in eV, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :param M_noCosmo_log10: it is \log_10 ( M/(D_lens^2 \Sigma_crit) ), M in M_sun, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :return: rho0 tilde = rho0 * D_lens / \Sigma_crit, central density in angular units
        """
        m22 = 10**(m_noCosmo_log10 +22)
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        return 190* 10**8 * m22**(-2) * rtilde**(-4)

    def lensing_Integral(self, x):
        """
        The analitic result of the integral entering the computation of the lensing potential

        :param x: evaluation point of the integral
        :return: result of the antiderivative in x
        """
        denominator = 3465*(x**2 +1)**(5.5)
        numerator = 3465*x**10 + 18480*x**8 + 39963*x**6 + 44154*x**4 + 25399*x**2 + 6508
        return np.log(np.sqrt(x**2 +1) + 1) - numerator/denominator

    def function(self, x, y, m_noCosmo_log10, M_noCosmo_log10, center_x=0, center_y=0):
        """
        lensing potential in arcsec^2
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential in arcsec^2
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_** 2 + y_** 2)
        r = np.maximum(r, self._s)
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        prefactor = 3.68 * rhotilde * rtilde**3
        result = prefactor * (self.lensing_Integral(np.sqrt(0.091)*r/theta_cArcsec) - self.lensing_Integral(0))
        # lensing potential has dimensions of angle^2
        return result/ const.arcsec**2

    def alpha_radial(self, r, m_noCosmo_log10, M_noCosmo_log10):
        """
        returns the radial part of the deflection angle

        :param r: radius where the deflection angle is computed
        :param m_noCosmo_log10: it is \log_10 (m \sqrt{\Sigma_crit D_lens^3} ), m in eV, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :param M_noCosmo_log10: it is \log_10 ( M/(D_lens^2 \Sigma_crit) ), M in M_sun, \Sigma_crit in M_sun / pc^2, D_lens in pc
        :return: radial deflection angle
        """
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        # Express r in arcsec and end with alpha in arcsec
        prefactor = 3.68 * rhotilde * rtilde**3 / const.arcsec**2
        denominator_factor = (1 + 0.091 * r**2/theta_cArcsec**2)**(6.5)
        return prefactor/r * (1 - 1/denominator_factor)

    def derivatives(self, x, y, m_noCosmo_log10, M_noCosmo_log10, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (lensing potential), which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        #  theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        f_x = self.alpha_radial(R, m_noCosmo_log10, M_noCosmo_log10) * x_ / R
        f_y = self.alpha_radial(R, m_noCosmo_log10, M_noCosmo_log10) * y_ / R
        return f_x, f_y

    def hessian(self, x, y, m_noCosmo_log10, M_noCosmo_log10, center_x=0, center_y=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        # Express r in arcsec and end with alpha in arcsec
        prefactor = 3.68 * rhotilde * rtilde**3 / const.arcsec**2
        # denominator factor
        denominator = 1 + 0.091 * R**2/theta_cArcsec**2
        factor1 = 1.183 * denominator**(-7.5) / (theta_cArcsec**2 * R**2)
        factor2 = 1/R**4 * (1 - denominator**(-6.5))
        f_xx = prefactor * (factor1 * x_**2 + factor2 * (y_**2 - x_**2))
        f_yy = prefactor * (factor1 * y_**2 + factor2 * (x_**2 - y_**2))
        f_xy = prefactor * (factor1 * x_ * y_ - factor2 * 2*x_*y_)
        return f_xx, f_yy, f_xy

    def density(self, R, m_noCosmo_log10, M_noCosmo_log10):
        """
        three dimensional ULDM profile in angular units (rho0_physical = rho0_angular \Sigma_crit / D_lens)

        :param R: radius of interest
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: rho(R) density in angular units
        """
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        return rhotilde/(1 + 0.091* (R/theta_cArcsec)**2)**8

    def density_lens(self, r, m_noCosmo_log10, M_noCosmo_log10):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: 3d radius
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: density rho(r)
        """
        return self.density(r, m_noCosmo_log10, M_noCosmo_log10)

    def density_2d(self, x, y, m_noCosmo_log10, M_noCosmo_log10, center_x=0, center_y=0):
        """
        projected two dimensional ULDM profile (convergence * \Sigma_crit), but given our
        units convention for rho0, it is basically the convergence

        :param R: radius of interest
        :type R: float/numpy array
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        # remember that the convergence is adimensional
        return 2.18 * rhotilde * rtilde / (1 + 0.091 * (R / theta_cArcsec)**2)**(7.5)

    def mass_Integral(self, x):
        """
        Returns the analitic result of the integral appearing in mass expression
        """
        numerator = x * (3465 * x**12 + 23100 * x**10 + 65373 * x**8 + 101376*x**6 + 92323*x**4 + 48580 * x**2 - 3465)
        denominator = 215040 * (x**2 + 1)**7
        result = 33 * np.arctan(x) / 2048 + numerator/denominator
        return result

    def mass_3d(self, R, m_noCosmo_log10, M_noCosmo_log10):
        """
        mass enclosed a 3d sphere or radius r
        :param R: radius in arcseconds
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: mass of soliton in angular units
        """
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        m_3d = 4. * np.pi * rhotilde * rtilde**3 / (0.091)**(1.5) * (self.mass_Integral(R/theta_cArcsec * np.sqrt(0.091)) - self.mass_Integral(0) )
        return m_3d

    def mass_3d_lens(self, r, m_noCosmo_log10, M_noCosmo_log10):
        """
        mass enclosed a 3d sphere or radius r
        :param R:
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param rho0: central density in angular units
        :return: mass
        """
        m_3d = self.mass_3d(r, m_noCosmo_log10, M_noCosmo_log10)
        return m_3d

    def mass_2d(self, R, m_noCosmo_log10, M_noCosmo_log10):
        """
        mass enclosed a 2d sphere or radius r
        :param R: radius over which the mass is computed
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: mass enclosed in 2d sphere
        """
        rtilde = self.theta_cRad(m_noCosmo_log10, M_noCosmo_log10)
        rhotilde = self.rho0Tilde(m_noCosmo_log10, M_noCosmo_log10)
        theta_cArcsec = rtilde / const.arcsec
        integral_factor = 1 - (1 + 0.091*(R/ theta_cArcsec)**2)**(-6.5)
        m_2d = 2*np.pi * rhotilde * rtilde**3 * 1.84 * integral_factor
        return m_2d

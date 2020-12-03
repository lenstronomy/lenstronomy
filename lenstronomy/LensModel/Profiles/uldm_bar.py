__author__ = 'lucateo'

# this file contains a class to compute the Ultra Light Dark Matter soliton profile
import numpy as np
import scipy.interpolate as interp
from scipy.special import exp1, erf
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const
from lenstronomy.LensModel.Profiles.spep import SPEP
from lenstronomy.LensModel.Profiles.spp import SPP

__all__ = ['ULDM-BAR']

class Uldm_Bar(LensProfileBase):
    """
    this class contains functions concerning the ULDM soliton density profile with baryons,
    modelled as power law
    """
    _s = 0.000001  # numerical limit for minimal radius
    _Spep = SPEP() # call softened elliptical power law profile
    param_names = ['kappa_0', 'theta_c', 'theta_E', 'gamma', 'e1', 'e2', 'center_xULDM', 'center_yULDM', 'center_x', 'center_y']
    lower_limit_default = {'kappa_0': 0, 'theta_c': 0, 'theta_E': 0, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_xULDM': -100, 'center_yULDM': -100, 'center_x': -100, 'center_y': -100 }
    upper_limit_default = {'kappa_0': 10, 'theta_c': 100, 'theta_E': 100, 'gamma': 100, 'e1': 0.5, 'e2': 0.5, 'center_xULDM': 100, 'center_yULDM': 100, 'center_x': 100, 'center_y': 100 }

    def rhoTilde(self, kappa_0, theta_c):
        """
        Computes the central density in angular units
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: central density in 1/arcsec
        """
        return kappa_0 / (np.sqrt(np.pi) * theta_c)

    def function(self, x, y, kappa_0, theta_c, theta_E, gamma, e1, e2, center_xULDM = 0, center_yULDM = 0, center_x=0, center_y=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential (in arcsec^2)
        """
        functionPL = self._Spep.function(x,y, theta_E, gamma, e1, e2, center_x, center_y)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_** 2 + y_** 2)
        r = np.maximum(r, self._s)
        Integral_factor = 0.5 * exp1( (r/theta_c)**2) + np.log( (r/theta_c))
        functionULDM = kappa_0 * theta_c**2 * Integral_factor
        return functionULDM + functionPL

    def alpha_radial(self, r, kappa_0, theta_c):
        """
        returns the radial part of the deflection angle for the ULDM profile only

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: radial deflection angle
        """
        prefactor = kappa_0 * theta_c**2 / r
        return prefactor * (1 - np.exp(- (r/theta_c)**2 ))

    def derivatives(self, x, y, kappa_0, theta_c, theta_E, gamma, e1, e2, center_xULDM = 0, center_yULDM = 0, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (lensing potential), which are the deflection angles

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :return: deflection angle in x, deflection angle in y
        """
        PL_fx, PL_fy = self._Spep.derivatives(x,y, theta_E, gamma, e1, e2, center_x, center_y)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        f_x = self.alpha_radial(R, kappa_0, theta_c ) * x_ / R
        f_y = self.alpha_radial(R, kappa_0, theta_c) * y_ / R
        return f_x + PL_fx, f_y + PL_fy

    def hessian(self, x, y, kappa_0, theta_c, theta_E, gamma, e1, e2, center_xULDM = 0, center_yULDM = 0, center_x=0, center_y=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        PL_fxx, PL_fyy, PL_fxy = self._Spep.hessian(x,y, theta_E, gamma, e1, e2, center_x, center_y)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        prefactor = np.sqrt(np.pi) * rhotilde * theta_c**3
        expFactor = np.exp( - (R/theta_c)**2)
        factor1 = (1 - expFactor)/R**4
        factor2 = 2/(R**2 * theta_c**2) * expFactor
        f_xx = prefactor * ( factor1 * (y_**2 - x_**2) + factor2 * x_**2 )
        f_yy = prefactor * ( factor1 * (x_**2 - y_**2) + factor2 * y_**2 )
        f_xy = prefactor * ( - factor1 * 2 * x_ * y_ + factor2 *x_*y_ )
        return f_xx+ PL_fxx, f_yy + PL_fyy, f_xy + PL_fxy

    def density(self, R, kappa_0, theta_c):
        """
        three dimensional ULDM profile in angular units (rho0_physical = rho0_angular \Sigma_crit / D_lens)

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: rho(R) density
        """
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        return rhotilde * np.exp( -(R/theta_c)**2)

    def density_lens(self, r, kappa_0, theta_c, theta_E, gamma, e1=0, e2=0):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :return: density rho(r)
        """
        PL_density = self._Spep.density_lens(r, theta_E, gamma, e1, e2)
        return self.density(r, kappa_0, theta_c) + PL_density

    def density_2d_ULDM(self, x, y, kappa_0, theta_c):
        """
        projected two dimensional ULDM profile (convergence * \Sigma_crit), but given our
        units convention for rho0, it is basically the convergence

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        expFactor = np.exp( - (R/theta_c)**2)
        return kappa_0  * expFactor

    def mass_3d(self, R, kappa_0, theta_c):
        """
        mass enclosed a 3d sphere or radius r for ULDM profile only
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param R: radius in arcseconds
        :return: mass of soliton in angular units
        """
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        integral_factor = np.sqrt(np.pi) * erf(R/theta_c)/2 - R* np.exp(-(R/theta_c)**2)
        m_3d =  2* np.pi * rhotilde * theta_c**3 * integral_factor
        return m_3d

    def mass_3d_lens(self, r, kappa_0, theta_c, theta_E, gamma, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: mass
        """
        m_3dPL = self._Spep.mass_3d_lens(r, theta_E, gamma, e1, e2)
        m_3d = self.mass_3d(r, kappa_0, theta_c)
        return m_3d + m_3dPL

    def mass_2d(self, R, kappa_0, theta_c):
        """
        mass enclosed a 2d sphere of radius r, ULDM profile only.
        returns M_2D = 2 \pi r \int dz \rho(\sqrt(r^2 + z^2))
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :return: M_2D (ULDM only)
        """
        exp_factor =  np.exp(-(R/theta_c)**2)
        rhotilde = self.rhoTilde(kappa_0, theta_c)
        m_2d = 2*np.pi**(1.5) * R * theta_c * rhotilde * exp_factor
        return m_2d

    def mass_2d_lens(self, R, kappa_0, theta_c, theta_E, gamma):
        """
        mass enclosed a 2d sphere of radius r, both ULDM and PL profiles.
        returns M_2D = 2 \pi r \int dz \rho(\sqrt(r^2 + z^2))
        :param kappa_0: central convergence of soliton
        :param theta_c: core radius (in arcsec)
        :param theta_E: PL Einstein angle
        :param gamma: PL slope
        :return: M_2D
        """
        m_2dULDM = self.mass_2d(R,kappa_0, theta_c)
        m_2dPL = SPP.mass_2d_lens(R, theta_E, gamma)
        return m_2dULDM + m_2dPL

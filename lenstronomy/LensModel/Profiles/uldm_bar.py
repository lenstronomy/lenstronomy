__author__ = 'lucateo'

# this file contains a class to compute the Ultra Light Dark Matter soliton profile
import numpy as np
import scipy.interpolate as interp
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import lenstronomy.Util.constants as const
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.hernquist import Hernquist

__all__ = ['ULDM-BAR']


class Uldm_Bar(LensProfileBase):
    """
    this class contains functions concerning the ULDM soliton density profile with baryons,
    modelled as Hernquist + NFW
    """
    _s = 0.000001  # numerical limit for minimal radius
    _NFW = NFW() # call NFW profile
    _Hernquist = Hernquist() # call Hernquist profile
    # NFW are NFW profile parameters, H are hernquist parameters
    param_names = ['m_noCosmo_log10', 'M_noCosmo_log10', 'RsNFW', 'alpha_RsNFW', 'sigma0H','RsH', 'center_x', 'center_y', 'center_xNFW', 'center_yNFW','center_xH', 'center_yH']
    lower_limit_default = {'m_noCosmo_log10': -12, 'M_noCosmo_log10': -15, 'RsNFW': 0, 'alpha_RsNFW': 0, 'sigma0H': 0, 'RsH': 0, 'center_x': -100, 'center_y': -100, 'center_xNFW': -100, 'center_yNFW': -100, 'center_xH': -100, 'center_yH': -100}
    upper_limit_default = {'m_noCosmo_log10': -3, 'M_noCosmo_log10': -6, 'RsNFW': 100, 'alpha_RsNFW': 10, 'sigma0H': 100, 'RsH': 100, 'center_x': 100, 'center_y': 100, 'center_xNFW': 100, 'center_yNFW': 100, 'center_xH': 100, 'center_yH': 100}

    #  def __init__(self, interpol=False, num_interp_X=1000, max_interp_X=10):
    #      """
    #
    #      #  :param interpol: bool, if True, interpolates the functions F(), g() and h()
    #      #  :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
    #      #  :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated (returning zeros outside)
    #      #  """
    #      super(Uldm, self).__init__()

    def mass2angles(self, m_noCosmo_log10, M_noCosmo_log10):
        """
        Function to pass from the m and M without cosmology to angles
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: theta_c, alpha_c
        """

        m22 = 10**(m_noCosmo_log10 + 22)
        M9 = 10**(M_noCosmo_log10 -9)
        theta_c = 227.78 * m22**(-2) * M9**(-1) / const.arcsec # In arcseconds
        # Multiply by arcsec for correct units (theta_c already in arcsec, so a theta_c^2 term
        # would give a arcsec^2 dimensions; remember that here const.arcsec = arcsec2rad and
        # both dd and sigma_crit use Mpc as length unit
        alpha_c = 1.59 * 1.9 * 10**(10) / 227.78**2 * m22**2 * M9**2 / const.arcsec # In arcseconds
        return theta_c, alpha_c

############## CHANGE
    def lensing_Integral(self, x):
        """
        The analitic result of the integral entering the computation of the lensing potential

        :param x: evaluation point of the integral
        :return: result of the antiderivative in x
        """
        denominator = 3465*(x**2 +1)**(5.5)
        numerator = 3465*x**10 + 18480*x**8 + 39963*x**6 + 44154*x**4 + 25399*x**2 + 6508
        return np.log(np.sqrt(x**2 +1) + 1) - numerator/denominator

    def function(self, x, y, m_noCosmo_log10, M_noCosmo_log10, RsNFW, alpha_RsNFW, sigma0H, RsH, center_x=0, center_y=0, center_xNFW=0, center_yNFW=0,center_xH=0, center_yH=0):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential
        """
        functionNFW = self._NFW.function(x,y,RsNFW, alpha_RsNFW, center_xNFW, center_yNFW)
        functionHernq = self._Hernquist.function(x, y, sigma0H, RsH, center_xH, center_yH)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_** 2 + y_** 2)
        r = np.maximum(r, self._s)
        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        result = alpha_c/0.43 * theta_c/r * (self.lensing_Integral(np.sqrt(0.091)*r/theta_c) - self.lensing_Integral(0))
        return result + functionNFW + functionHernq

    def alpha_radial(self, r, alpha_c, theta_c):
        """
        returns the radial part of the deflection angle

        :param r: radius where the deflection angle is computed
        :param alpha_c: deflection angle at theta_c
        :param theta_c: core radius of ULDM soliton
        :return: radial deflection angle
        """
        prefactor = alpha_c/0.43 * theta_c
        denominator_factor = (1 + 0.091 * r**2/theta_c**2)**(6.5)
        return prefactor/r * (1 - 1/denominator_factor)

    def derivatives(self, x, y, m_noCosmo_log10, M_noCosmo_log10, RsNFW, alpha_RsNFW, sigma0H, RsH, center_x=0, center_y=0, center_xNFW=0, center_yNFW=0,center_xH=0, center_yH=0):
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
        NFW_fx, NFW_fy = self._NFW.derivatives(x,y,RsNFW, alpha_RsNFW, center_xNFW, center_yNFW)
        Hernq_fx, Hernq_fy = self._Hernquist.derivatives(x, y, sigma0H, RsH, center_xH, center_yH)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        f_x = self.alpha_radial(R, alpha_c, theta_c) * x_ / R
        f_y = self.alpha_radial(R, alpha_c, theta_c) * y_ / R
        return f_x + NFW_fx + Hernq_fx, f_y + NFW_fy + Hernq_fy

    def hessian(self, x, y, m_noCosmo_log10, M_noCosmo_log10, RsNFW, alpha_RsNFW, sigma0H, RsH, center_x=0, center_y=0, center_xNFW=0, center_yNFW=0,center_xH=0, center_yH=0):
        """
        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        NFW_fxx, NFW_fyy, NFW_fxy = self._NFW.hessian(x,y,RsNFW, alpha_RsNFW, center_xNFW, center_yNFW)
        Hernq_fxx, Hernq_fyy, Hernq_fxy = self._Hernquist.hessian(x, y, sigma0H, RsH, center_xH, center_yH)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R,0.00000001)
        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        factor1 = 2.74 * alpha_c / theta_c * (1 + 0.091*R**2 / theta_c**2)**(-7.5)/R**2
        factor2 = 1/R**4 * (1 - (1 + 0.091 * R**2 / theta_c**2)**(-6.5))
        f_xx = factor1 * x_**2 + factor2 * (y_**2 - x_**2)
        f_yy = factor1 * y_**2 + factor2 * (x_**2 - y_**2)
        f_xy = factor1 * x_ * y_ - factor2 * 2*x_*y_
        return f_xx+ NFW_fxx + Hernq_fxx, f_yy + NFW_fyy + Hernq_fyy, f_xy + NFW_fxy + Hernq_fxy

    def density(self, R, rho0, theta_c):
        """
        three dimensional ULDM profile in angular units (rho0_physical = rho0_angular \Sigma_crit / D_lens)

        :param R: radius of interest
        :type R: float/numpy array
        :param rho0: central density in angular units
        :type rho0: float
        :param theta_c: core angle
        :type theta_c: float
        :return: rho(R) density
        """
        return rho0/(1 + 0.091* (R/theta_c)**2)**8

    def alpha_c2rho0(self, theta_c, alpha_c):
        """
        Converts deflection angle at the core radius in central density in angular units

        :param alpha_c: deflection angle at theta_c
        :param theta_c: core radius of ULDM soliton
        :return: rho0
        """
        return alpha_c / theta_c**2 / 1.59

    def density_lens(self, r, m_noCosmo_log10, M_noCosmo_log10, RsNFW, alpha_RsNFW, sigma0H, RsH):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: 3d radius
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :return: density rho(r)
        """
        NFW_density = self._NFW.density_lens(r,RsNFW, alpha_RsNFW)
        Hernq_density = self._Hernquist.density_lens(r, sigma0H, RsH)
        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        rho0 = self._alpha_c2rho0(theta_c, alpha_c)
        return self.density(r, rho0, theta_c) + NFW_density + Hernq_density

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
        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        R = np.sqrt(x_**2 + y_**2)
        rho0 = self.alpha_c2rho0(theta_c,alpha_c)
        return 2.18 * rho0 * theta_c / (1 + 0.091 * (R / theta_c)**2)**(7.5)

    def mass_Integral(self, x):
        """
        Returns the analitic result of the integral appearing in mass expression
        """
        numerator = x * (3465 * x**12 + 23100 * x**10 + 65373 * x**8 + 101376*x**6 + 92323*x**4 + 48580 * x**2 - 3465)
        denominator = 215040 * (x**2 + 1)**7
        result = 33 * np.arctan(x) / 2048 + numerator/denominator
        return result

    def mass_3d(self, R, theta_c, rho0):
        """
        mass enclosed a 3d sphere or radius r
        :param R: radius in arcseconds
        :param theta_c: angle at the core
        :param rho0: central density in angular units
        :return: mass of soliton in angular units
        """
        m_3d = 4. * np.pi * rho0 * theta_c**3 / (0.091)**(1.5) * (self.mass_Integral(R/theta_c * np.sqrt(0.091)) - self.mass_Integral(0) )
        return m_3d

    def mass_3d_lens(self, r, m_noCosmo_log10, M_noCosmo_log10, RsNFW, alpha_RsNFW, sigma0H, RsH):
        """
        mass enclosed a 3d sphere or radius r
        :param R:
        :param m_noCosmo_log10: m \sqrt{\Sigma_crit D_lens^3}, mass in eV, Sigma_crit in M_sun/parsec^2
        :param M_noCosmo_log10: M/(D_lens^2 \Sigma_crit), M in M_sun, Sigma_crit in M_sun/parsec^2
        :param rho0: central density in angular units
        :return: mass
        """

        theta_c, alpha_c = self.mass2angles(m_noCosmo_log10, M_noCosmo_log10)
        rho0 = self._alpha_c2rho0(theta_c, alpha_c)
        m_3d = self.mass_3d(r, theta_c, rho0)
        return m_3d

    def mass_2d(self, R, theta_c, rho0):
        """
        mass enclosed a 2d sphere or radius r
        :param r:
        :param theta_c: angle at the core
        :param rho0: central density in angular units
        :return:
        """
        integral_factor = 1 - (1 + 0.091*(R/ theta_c)**2)**(-6.5)
        m_2d = 2*np.pi * rho0 * theta_c**3 * 1.84 * integral_factor
        return m_2d


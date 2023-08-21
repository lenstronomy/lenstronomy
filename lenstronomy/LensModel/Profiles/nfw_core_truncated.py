__author__ = 'dgilman'

# this file contains a class to compute lensing proprerties of a pseudo Navaro-Frenk-White profile with a core and truncation
# radius
import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from scipy.integrate import quad

__all__ = ['TNFWC']


class TNFWC(LensProfileBase):
    """This class contains an pseudo NFW profile with a core radius and a truncation
    radius. The density in 3D is given by.

    .. math::
        \\rho(r) = \\frac{\\rho_0 r_s^3}{\\left(r^2+r_c^2\\right)^{1/2} \\left(r_s^2+r^2\\right)} \\left(\\frac{r_t^2}{r^2+r_t^2}\\right)

    When the core radius goes to zero and the truncation radius approaches infinity this profile reduces to an NFW profile
    with the squared term inside the parentheses.

    TODO: add the gravitational potential for this profile
    TODO: add analytic solution for 3D mass
    """
    profile_name = 'TNFWC'
    param_names = ['Rs', 'alpha_Rs', 'center_x', 'center_y', 'r_trunc', 'r_core']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'center_x': -100, 'center_y': -100, 'r_trunc': 0.001,
                           'r_core': 0.00001}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'center_x': 100, 'center_y': 100, 'r_trunc': 1000.0,
                           'r_core': 1000.0}

    def derivatives(self, x, y, Rs, alpha_Rs, r_core, r_trunc, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function which are the deflection angles.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_input = self.alpha2rho0(alpha_Rs, Rs, r_core, r_trunc)
        Rs = np.maximum(Rs, 0.00000001)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        f_x, f_y = self.nfw_alpha(R, Rs, rho0_input, r_core, r_trunc, x_, y_)
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, r_core, r_trunc, center_x=0, center_y=0):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        rho0_input = self.alpha2rho0(alpha_Rs, Rs, r_core, r_trunc)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        R = np.maximum(R, 0.00000001)
        kappa = self.density_2d(R, 0, Rs, rho0_input, r_core, r_trunc)
        gamma1, gamma2 = self.nfw_gamma(R, Rs, rho0_input, r_core, r_trunc, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def density(R, Rs, rho0, r_core, r_trunc):
        """3D density profile.

        :param R: radius of interest
        :type Rs: scale radius
        :param rho0: central density normalization
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: rho(R) density
        """
        x = R / Rs
        beta = r_core / Rs
        tau = r_trunc / Rs
        denom_core = (beta ** 2 + x ** 2) ** 0.5
        denom_nfw = (1 + x ** 2)
        denom_trunc = (x ** 2 + tau ** 2) / tau ** 2
        denom = denom_core * denom_nfw * denom_trunc
        return rho0 / denom

    def density_lens(self, r, Rs, alpha_Rs, r_core, r_trunc):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: 3d radios
        :param Rs: scale radius
        :param alpha_Rs: deflection at Rs
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: density rho(r)
        """
        rho0 = self.alpha2rho0(alpha_Rs, Rs, r_core, r_trunc)
        return self.density(r, Rs, rho0, r_core, r_trunc)

    def density_2d(self, x, y, Rs, rho0, r_core, r_trunc, center_x=0, center_y=0):
        """2D (projected) density profile.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param rho0: density normalization at Rs
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :param center_x: profile center (same units as x)
        :param center_y: profile center (same units as x)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_ ** 2 + y_ ** 2)
        x = R / Rs
        beta = r_core / Rs
        tau = r_trunc / Rs
        Fx = self._f(x, beta, tau)
        return 2 * rho0 * Rs * Fx

    def mass_3d(self, r, Rs, rho0, r_core, r_trunc):
        """Mass enclosed a 3d sphere or radius r.

        :param r: 3d radius
        :param Rs: scale radius
        :param rho0: density normalization
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: M(<r)
        """
        integrand = lambda x: x ** 2 * self.density(x, Rs, rho0, r_core, r_trunc)
        return 4 * np.pi * quad(integrand, 0, r)[0]

    def mass_3d_lens(self, r, Rs, alpha_Rs, r_core, r_trunc):
        """Mass enclosed a 3d sphere or radius r. This function takes as input the
        lensing parameterization.

        :param r: 3d radius
        :param Rs: scale radius
        :param alpha_Rs: deflection angle at Rs
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: M(<r)
        """
        rho0 = self.alpha2rho0(alpha_Rs, Rs, r_core, r_trunc)
        m_3d = self.mass_3d(r, Rs, rho0, r_core, r_trunc)
        return m_3d

    def mass_2d(self, R, Rs, rho0, r_core, r_trunc):
        """Mass enclosed a 2d cylinder or projected radius R.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: mass in cylinder
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        beta = r_core / Rs
        tau = r_trunc / Rs
        gx = self._g(x, beta, tau)
        m_2d = 4 * rho0 * Rs * R ** 2 * gx / x ** 2 * np.pi
        return m_2d

    def nfw_alpha(self, R, Rs, rho0, r_core, r_trunc, ax_x, ax_y):
        """Deflection angle of the profile (times Sigma_crit D_OL) along the projection
        to coordinate 'axis'.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :param ax_x: x coordinate relative to center
        :param ax_y: y coordinate relative to center
        :return: Epsilon(R) projected density at radius R
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        beta = r_core / Rs
        tau = r_trunc / Rs
        gx = self._g(x, beta, tau)
        a = 4 * rho0 * Rs * R * gx / x ** 2 / R
        return a * ax_x, a * ax_y

    def nfw_gamma(self, R, Rs, rho0, r_core, r_trunc, ax_x, ax_y):
        """Shear gamma of NFW profile (times Sigma_crit) along the projection to
        coordinate 'axis'.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :param ax_x: x coordinate relative to center
        :param ax_y: y coordinate relative to center
        :return: Epsilon(R) projected density at radius R
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        beta, tau = r_core / Rs, r_trunc / Rs
        gx = self._g(x, beta, tau)
        Fx = self._f(x, beta, tau)
        a = 2 * rho0 * Rs * (2 * gx / x ** 2 - Fx)  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a * (ax_y ** 2 - ax_x ** 2) / R ** 2, -a * 2 * (ax_x * ax_y) / R ** 2

    def _f(self, x, b, t):
        """Analytic solution of the projection integral.

        :param X: R/Rs
        :type X: float >0
        :param b: core radius divided by the scale radius
        :param t: truncation radius divided by the scale radius
        :return: solution to the projection integral
        """
        prefactor = t ** 2 / (t ** 2 - 1)
        return prefactor * (self._u1(x, b, 1.0) - self._u1(x, b, t))

    def _g(self, x, b, t):
        """Analytic solution of integral for NFW profile to compute deflection angle and
        gamma.

        :param X: R/Rs
        :type X: float >0
        :param b: core radius divided by the scale radius
        :param t: truncation radius divided by the scale radius
        :return: solution of the integral over projected mass
        """
        if b == t:
            t += 1e-3
        prefactor = abs(t ** 2 / (t ** 2 - 1))
        return prefactor * (-self._u2(x, b, t) + self._u2(0.0, b, t) + self._u2(x, b, 1.0) - self._u2(0.0, b, 1.0))

    @staticmethod
    def _u1(x, b, t):
        """
        :param x: R/Rs
        :param b: core radius divided by the scale radius
        :param t: truncation radius divided by the scale radius

        """
        t2x2 = t ** 2 + x ** 2
        b2x2 = b ** 2 + x ** 2
        b2mt2 = b ** 2 - t ** 2
        if t > b:
            func = np.arccosh
            b2mt2 *= -1
        else:
            func = np.arccos
        arg = np.sqrt(t2x2 / b2x2)
        return func(arg) / np.sqrt(t2x2 * b2mt2)

    def _u2(self, x, b, t):
        """
        :param x: R/Rs
        :param b: core radius divided by the scale radius
        :param t: truncation radius divided by the scale radius

        """
        return (t ** 2 + x ** 2) * self._u1(x, b, t)

    def alpha2rho0(self, alpha_Rs, Rs, r_core, r_trunc):
        """Convert angle at Rs into rho0.

        :param alpha_Rs: deflection angle at RS
        :param Rs: scale radius
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: density normalization
        """
        # alpha_Rs = 4 * rho0 * Rs * R * gx
        beta = r_core / Rs
        tau = r_trunc / Rs
        gx = self._g(1.0, beta, tau)
        rho0 = alpha_Rs / (4 * Rs ** 2 * gx)
        return rho0

    def rho02alpha(self, rho0, Rs, r_core, r_trunc):
        """Convert rho0 to angle at Rs.

        :param rho0: density normalization
        :param Rs: scale radius
        :param r_core: core radius [arcsec]
        :param r_trunc: truncation radius [arcsec]
        :return: deflection angle at RS
        """
        return self.nfw_alpha(Rs, Rs, rho0, r_core, r_trunc, Rs, 0.0)[0]

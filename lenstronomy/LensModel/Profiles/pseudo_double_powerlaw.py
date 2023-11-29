__author__ = "dgilman"

import numpy as np
from scipy.special import hyp2f1
from scipy.special import beta
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["PSEUDO_DPL"]


class PseudoDoublePowerlaw(LensProfileBase):
    """This class contains a double power law profile with flexible inner and outer
    logarithmic slopes g and n.

    .. math::
        \\rho(r) = \\frac{\\rho_0}{r^{\\gamma}} \\frac{Rs^{n}}{\\left(r^2 + Rs^2 \\right)^{(n - \\gamma)/2}}

    For g = 1.0 and n=3, it is approximately the same as an NFW profile
    The original reference is [1]_.

    .. [1] Munoz, Kochanek and Keeton, (2001), astro-ph/0103009, doi:10.1086/322314

    TODO: implement the gravitational potential for this profile
    """

    profile_name = "PSEUDO_DPL"
    param_names = [
        "Rs",
        "alpha_Rs",
        "center_x",
        "center_y",
        "gamma_inner",
        "gamma_outer",
    ]
    lower_limit_default = {
        "Rs": 0,
        "alpha_Rs": 0,
        "center_x": -100,
        "center_y": -100,
        "gamma_inner": 0.1,
        "gamma_outer": 1.0,
    }
    upper_limit_default = {
        "Rs": 100,
        "alpha_Rs": 10,
        "center_x": 100,
        "center_y": 100,
        "gamma_inner": 2.9,
        "gamma_outer": 10.0,
    }

    def derivatives(
        self, x, y, Rs, alpha_Rs, gamma_inner, gamma_outer, center_x=0, center_y=0
    ):
        """Returns df/dx and df/dy of the function which are the deflection angles.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection angle in x, deflection angle in y
        """
        rho0_input = self.alpha2rho0(alpha_Rs, Rs, gamma_inner, gamma_outer)
        Rs = np.maximum(Rs, 0.00000001)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        f_x, f_y = self.alpha(R, Rs, rho0_input, gamma_inner, gamma_outer, x_, y_)
        return f_x, f_y

    def hessian(
        self, x, y, Rs, alpha_Rs, gamma_inner, gamma_outer, center_x=0, center_y=0
    ):
        """

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        rho0_input = self.alpha2rho0(alpha_Rs, Rs, gamma_inner, gamma_outer)
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R, 0.00000001)
        kappa = self.density_2d(R, 0, Rs, rho0_input, gamma_inner, gamma_outer)
        gamma1, gamma2 = self.gamma(R, Rs, rho0_input, gamma_inner, gamma_outer, x_, y_)
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_xy, f_yy

    @staticmethod
    def density(R, Rs, rho0, gamma_inner, gamma_outer):
        """Three dimensional NFW profile.

        :param R: radius of interest
        :type Rs: scale radius
        :param rho0: central density normalization
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: rho(R) density
        """
        x = R / Rs
        outer_slope = (gamma_outer - gamma_inner) / 2
        return rho0 / (x**gamma_inner * (1 + x**2) ** outer_slope)

    def density_lens(self, r, Rs, alpha_Rs, gamma_inner, gamma_outer):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param r: 3d radios
        :param Rs: scale radius
        :param alpha_Rs: deflection at Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: density rho(r)
        """
        rho0 = self.alpha2rho0(alpha_Rs, Rs, gamma_inner, gamma_outer)
        return self.density(r, Rs, rho0, gamma_inner, gamma_outer)

    def density_2d(
        self, x, y, Rs, rho0, gamma_inner, gamma_outer, center_x=0, center_y=0
    ):
        """Projected two dimensional profile.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param rho0: density normalization at Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :param center_x: profile center (same units as x)
        :param center_y: profile center (same units as x)
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        x = R / Rs
        Fx = self._f(x, gamma_inner, gamma_outer)
        return 2 * rho0 * Rs * Fx

    @staticmethod
    def mass_3d(r, Rs, rho0, gamma_inner, gamma_outer):
        """Mass enclosed a 3d sphere or radius r.

        :param r: 3d radius
        :param Rs: scale radius
        :param rho0: density normalization
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: M(<r)
        """
        Rs = float(Rs)
        const = 4 * np.pi * r**3 * rho0 * (Rs / r) ** gamma_inner
        m_3d = (
            const
            / (3 - gamma_inner)
            * hyp2f1(
                (3 - gamma_inner) / 2,
                (gamma_outer - gamma_inner) / 2,
                (5 - gamma_inner) / 2,
                -((r / Rs) ** 2),
            )
        )
        return m_3d

    def mass_3d_lens(self, r, Rs, alpha_Rs, gamma_inner, gamma_outer):
        """Mass enclosed a 3d sphere or radius r. This function takes as input the
        lensing parameterization.

        :param r: 3d radius
        :param Rs: scale radius
        :param alpha_Rs: deflection angle at Rs
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: M(<r)
        """
        rho0 = self.alpha2rho0(alpha_Rs, Rs, gamma_inner, gamma_outer)
        m_3d = self.mass_3d(r, Rs, rho0, gamma_inner, gamma_outer)
        return m_3d

    def mass_2d(self, R, Rs, rho0, gamma_inner, gamma_outer):
        """Mass enclosed a 2d cylinder or projected radius R.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: mass in cylinder
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        gx = self._g(x, gamma_inner, gamma_outer)
        m_2d = 4 * rho0 * Rs * R**2 * gx / x**2 * np.pi
        return m_2d

    def alpha(self, R, Rs, rho0, gamma_inner, gamma_outer, ax_x, ax_y):
        """Deflection angel of NFW profile (times Sigma_crit D_OL) along the projection
        to coordinate 'axis'.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :param ax_x: x coordinate relative to center
        :param ax_y: y coordinate relative to center
        :return: Epsilon(R) projected density at radius R
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        gx = self._g(x, gamma_inner, gamma_outer)
        a = 4 * rho0 * Rs * R * gx / x**2 / R
        return a * ax_x, a * ax_y

    def gamma(self, R, Rs, rho0, gamma_inner, gamma_outer, ax_x, ax_y):
        """Shear gamma of NFW profile (times Sigma_crit) along the projection to
        coordinate 'axis'.

        :param R: 3d radius
        :param Rs: scale radius
        :param rho0: central density normalization
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :param ax_x: x coordinate relative to center
        :param ax_y: y coordinate relative to center
        :return: Epsilon(R) projected density at radius R
        """
        R = np.maximum(R, 0.00000001)
        x = R / Rs
        gx = self._g(x, gamma_inner, gamma_outer)
        Fx = self._f(x, gamma_inner, gamma_outer)
        a = (
            2 * rho0 * Rs * (2 * gx / x**2 - Fx)
        )  # /x #2*rho0*Rs*(2*gx/x**2 - Fx)*axis/x
        return a * (ax_y**2 - ax_x**2) / R**2, -a * 2 * (ax_x * ax_y) / R**2

    @staticmethod
    def _f(X, g, n):
        """Analytic solution of the projection integral.

        :param X: R/Rs
        :type X: float >0
        :param g: logarithmic profile slope interior to Rs
        :param n: logarithmic profile slope exterior to Rs
        :return: solution to the projection integral
        """
        if n == 3:
            n = 3.001  # for numerical stability
        hyp2f1_term = hyp2f1((n - 1) / 2, g / 2, n / 2, 1 / (1 + X**2))
        beta_term = beta((n - 1) / 2, 0.5)
        return 0.5 * beta_term * hyp2f1_term * (1 + X**2) ** ((1 - n) / 2)

    @staticmethod
    def _g(X, g, n):
        """Analytic solution of integral for NFW profile to compute deflection angel and
        gamma.

        :param X: R/Rs
        :type X: float >0
        :param g: logarithmic profile slope interior to Rs
        :param n: logarithmic profile slope exterior to Rs
        :return: solution of the integral over projected mass
        """
        if n == 3:
            n = 3.001  # for numerical stability
        xi = 1 + X**2
        hyp2f1_term = hyp2f1((n - 3) / 2, g / 2, n / 2, 1 / xi)
        beta_term_1 = beta((n - 3) / 2, (3 - g) / 2)
        beta_term_2 = beta((n - 3) / 2, 1.5)
        return 0.5 * (beta_term_1 - beta_term_2 * hyp2f1_term * xi ** ((3 - n) / 2))

    def alpha2rho0(self, alpha_Rs, Rs, gamma_inner, gamma_outer):
        """Convert angle at Rs into rho0.

        :param alpha_Rs: deflection angle at RS
        :param Rs: scale radius
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: density normalization (characteristic density)
        """

        gx = self._g(1.0, gamma_inner, gamma_outer)
        rho0 = alpha_Rs / (4.0 * Rs**2 * gx / 1.0**2)
        return rho0

    def rho02alpha(self, rho0, Rs, gamma_inner, gamma_outer):
        """Convert rho0 to angle at Rs.

        :param rho0: density normalization (characteristic density)
        :param Rs: scale radius
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: deflection angle at RS
        """
        gx = self._g(1.0, gamma_inner, gamma_outer)
        alpha_Rs = rho0 * (4.0 * Rs**2 * gx / 1.0**2)
        return alpha_Rs

__author__ = "ajshajib", "dgilman", "sibirrer"

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["GNFW"]


class GNFW(LensProfileBase):
    """
    This class computes the lensing quantities of a generalized NFW profile:

    .. math::
        \\rho(r) = \\frac{\\rho_{\\rm s}} { (r/r_{\\rm s}})^{\\gamma_{\\rm in}} * (1 + r/r_{\\rm
        s})^{3 - {\\gamma_{\\rm in}}}

    This class uses the normalization parameter `kappa_s` defined as:

    .. math::
        kappas_{\\rm s} = \\frac{\\rho_{\\rm s} r_{\\rm s}}{\\Sigma_{\\rm crit}}

    Some expressions are obtained from Keeton 2001
    https://ui.adsabs.harvard.edu/abs/2001astro.ph..2341K/abstract. See and cite the
    references therein.
    """

    model_name = "GNFW"
    _s = 0.001  # numerical limit for minimal radius
    param_names = ["Rs", "kappa_s", "gamma_in", "center_x", "center_y"]
    lower_limit_default = {
        "Rs": 0,
        "kappa_s": 0,
        "gamma_in": 0.0,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "Rs": 100,
        "kappa_s": 1000.0,
        "gamma_in": 3.0,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, trapezoidal_integration=False, integration_steps=1000):
        """

        :param trapezoidal_integrate: bool, if True, the numerical integral is performed
         with the trapezoidal rule, otherwise with ~scipy.integrate.quad
        :param integration_steps: number of steps in the trapezoidal integral
        """
        super(GNFW, self).__init__()
        self._integration_steps = integration_steps
        if trapezoidal_integration:
            self._integrate = self._trapezoidal_integrate
        else:
            self._integrate = self._quad_integrate

    def function(self, x, y, Rs, kappa_s, gamma_in, center_x=0, center_y=0):
        """Potential of gNFW profile.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: potential at radius r
        :rtype: float
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        r = np.maximum(r, self._s)

        if isinstance(r, int) or isinstance(r, float):
            return self._num_integral_potential(r, Rs, kappa_s, gamma_in)
        else:
            # TODO: currently the numerical integral is done one by one. More efficient is sorting the radial list and
            # then perform one numerical integral reading out to the radial points
            f_ = []
            for _r in r:
                f_.append(self._num_integral_potential(_r, Rs, kappa_s, gamma_in))
            return np.array(f_)

    def _num_integral_potential(self, r, Rs, kappa_s, gamma_in):
        """Compute the numerical integral of the potential.

        :param r: radius of interest
        :type r: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param kappa_s: convergence correspoding to rho0
        :return: potential at radius r
        :rtype: float
        """

        def _integrand(x):
            return self.alpha(x, Rs, kappa_s, gamma_in)

        return quad(_integrand, a=0, b=r)[0]

    def derivatives(self, x, y, Rs, kappa_s, gamma_in, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: deflection angle in x, deflection angle in y
        :rtype: float, float
        """
        if Rs < 0.0000001:
            Rs = 0.0000001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        R = np.maximum(R, self._s)

        f_r = self.alpha(R, Rs, kappa_s, gamma_in)
        f_x = f_r * x_ / R
        f_y = f_r * y_ / R

        return f_x, f_y

    def hessian(self, x, y, Rs, kappa_s, gamma_in, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy.

        :param x: angular position
        :type x: float/numpy array
        :param y: angular position
        :type y: float/numpy array
        :param Rs: angular turn over point
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: f_xx, f_xy, f_xy, f_yy
        :rtype: float, float, float, float
        """
        if Rs < 0.0001:
            Rs = 0.0001
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)

        kappa = self.kappa(R, Rs, kappa_s, gamma_in)
        f_r = self.alpha(R, Rs, kappa_s, gamma_in)
        f_rr = 2 * kappa - f_r / R

        cos_t = x_ / R
        sin_t = y_ / R

        f_xx = cos_t**2 * f_rr + sin_t**2 / R * f_r
        f_yy = sin_t**2 * f_rr + cos_t**2 / R * f_r
        f_xy = cos_t * sin_t * f_rr - cos_t * sin_t / R * f_r

        return f_xx, f_xy, f_xy, f_yy

    def density(self, R, Rs, rho0, gamma_in):
        """Three dimensional truncated NFW profile.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho(R) density
        :rtype: float
        """
        return rho0 * (R / Rs) ** -gamma_in * (1 + R / Rs) ** (gamma_in - 3)

    def density_lens(self, R, Rs, kappa_s, gamma_in):
        """Computes the density at 3d radius r given lens model parameterization. The
        integral in the LOS projection of this quantity results in the convergence
        quantity.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: density at radius R
        :rtype: float
        """
        rho0 = self.kappa_s2rho0(kappa_s=kappa_s, Rs=Rs, gamma_in=gamma_in)
        return self.density(R, Rs, rho0, gamma_in)

    def density_2d(self, x, y, Rs, rho0, gamma_in, center_x=0, center_y=0):
        """Projected two dimenstional NFW profile (kappa*Sigma_crit)

        :param x: x-coordinate
        :type x: float/numpy array
        :param y: y-coordinate
        :type y: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param center_x: center of halo
        :type center_x: float
        :param center_y: center of halo
        :type center_y: float
        :return: Epsilon(R) projected density at radius R
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        kappa_s = self.rho02kappa_s(rho0, Rs, gamma_in)

        return self.kappa(R, Rs, kappa_s, gamma_in)

    def mass_3d(self, R, Rs, rho0, gamma_in):
        """Mass enclosed a 3d sphere or radius r.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: mass enclosed a 3d sphere or radius r
        :rtype: float
        """
        M_0 = 4 * np.pi * rho0 * Rs**3 / (3 - gamma_in)
        x = R / Rs
        return (
            M_0
            * x ** (3 - gamma_in)
            * hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -x)
        )

    def mass_3d_lens(self, R, Rs, kappa_s, gamma_in):
        """Mass enclosed a 3d sphere or radius r given a lens parameterization with
        angular units.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: mass enclosed a 3d sphere or radius r
        :rtype: float
        """
        rho0 = self.kappa_s2rho0(kappa_s=kappa_s, Rs=Rs, gamma_in=gamma_in)
        return self.mass_3d(R, Rs, rho0, gamma_in)

    def _trapezoidal_integrate(self, func, x, gamma_in):
        """Integrate a function using the trapezoid rule.

        :param func: function to integrate
        :type func: function
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param steps: number of steps
        :type steps: int
        :return: integral
        :rtype: float
        """
        steps = self._integration_steps
        y = np.linspace(1e-10, 1 - 1e-10, steps)
        dy = y[1] - y[0]

        weights = np.ones(steps)
        weights[0] = 0.5
        weights[-1] = 0.5

        if isinstance(x, int) or isinstance(x, float):
            integral = np.sum(func(y, x, gamma_in) * dy * weights)
        else:
            ys = np.repeat(y[:, np.newaxis], len([x]), axis=1)

            integral = np.sum(
                func(ys, x, gamma_in) * dy * weights[:, np.newaxis], axis=0
            )

        return integral

    def _quad_integrate(self, func, x, gamma_in):
        """Integrate a function using the trapezoid rule.

        :param func: function to integrate
        :type func: function
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :param steps: number of steps
        :type steps: int
        :return: integral
        :rtype: float
        """
        if isinstance(x, int) or isinstance(x, float):
            integral = quad(func, a=0, b=1, args=(x, gamma_in))[0]
        else:
            integral = np.zeros_like(x)

            for i in range(len(x)):
                integral[i] = quad(func, a=0, b=1, args=(x[i], gamma_in))[0]

        return integral

    def _alpha_integrand(self, y, x, gamma_in):
        """Integrand of the deflection angel integral.

        :param y: integration variable
        :type y: np.array
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: integrand of the deflection angel integral
        """
        return (y + x) ** (gamma_in - 3) * (1 - np.sqrt(1 - y**2)) / y

    def _kappa_integrand(self, y, x, gamma_in):
        """Integrand of the deflection angel integral in eq. (57) of Keeton 2001.

        :param y: integration variable
        :type y: np.array
        :param x: x = R/Rs
        :type x: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: integrand of the deflection angel integral
        """
        return (y + x) ** (gamma_in - 4) * (1 - np.sqrt(1 - y**2))

    def alpha(self, R, Rs, kappa_s, gamma_in):
        """Deflection angel of gNFW profile along the radial direction.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: deflection angel at radius R
        :rtype: float
        """
        # R = np.maximum(R, self._s)
        x = R / Rs
        x = np.maximum(x, self._s)

        integral = self._integrate(self._alpha_integrand, x, gamma_in)

        alpha = (
            4
            * kappa_s
            * Rs
            * x ** (2 - gamma_in)
            * (
                hyp2f1(3 - gamma_in, 3 - gamma_in, 4 - gamma_in, -x) / (3 - gamma_in)
                + integral
            )
        )

        return alpha

    def kappa(self, R, Rs, kappa_s, gamma_in):
        """Convergence of gNFW profile along the radial direction.

        :param R: radius of interest
        :type R: float/numpy array
        :param Rs: scale radius
        :type Rs: float
        :param kappa_s: convergence correspoding to rho0
        :type kappa_s: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: convergence at radius R
        :rtype: float
        """
        x = R / Rs
        x = np.maximum(x, self._s)

        integral = self._integrate(self._kappa_integrand, x, gamma_in)

        kappa = (
            2
            * kappa_s
            * Rs
            * x ** (1 - gamma_in)
            * ((1 + x) ** (gamma_in - 3) + (3 - gamma_in) * integral)
        )

        return kappa

    @staticmethod
    def rho02kappa_s(rho0, Rs, gamma_in):
        """Convenience function to compute rho0 from alpha_Rs.

        :param rho0: density normalization (characteristic density)
        :type rho0: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: kappa_s
        :rtype: float
        """
        return rho0 * Rs

    @staticmethod
    def kappa_s2rho0(kappa_s, Rs, gamma_in):
        """Convenience function to compute rho0 from kappa_s. The returned rho_0 is
        normalized with $\\Sigma_{\\rm crit}$.

        :param kappa_s: convergence corresponding to rho0
        :type kappa_s: float
        :param Rs: scale radius
        :type Rs: float
        :param gamma_in: inner slope
        :type gamma_in: float
        :return: rho0
        :rtype: float
        """
        return kappa_s / Rs

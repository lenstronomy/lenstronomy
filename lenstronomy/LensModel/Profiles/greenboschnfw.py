__author__ = "jtekverk"

import numpy as np
from scipy.integrate import quad
from lenstronomy.LensModel.Profiles.radial_interpolated import RadialInterpolate
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["GreenBoschNFW"]


class GreenBoschNFW(LensProfileBase):
    """
    This class computes the lensing quantities of a tidally evolved NFW profile:

    .. math::
        \\rho(r,t) = \\frac{ f_{te} \\rho_{0} }
        { ( 1 + (\\frac{r}{r_s}\\frac{c_s - r_{te}}{c_s * r_{te}} )^{\\delta}) (\\frac{r}{r_s}) (1 + \\frac{r}{r_s})^2 }

    This class uses the dimensionless NFW normalization parameter "rho0ang" defined as:

    .. math::
        \\rho0ang = \\frac{ D_{l} \\rho_{0,phys} }{ \\Sigma_{crit} } ([Mpc] * [M_{solar}/Mpc^3 ] / [M_{solar}/Mpc^2] * [pi/180/3600 radians/arcsecond]),
        where D_{l} is the angular diameter distance to the lens in Mpc

    The density profile is defined in Green/Bosch 2019, see: https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.2091G/abstract
    """

    model_name = "GreenBoschNFW"
    param_names = [
        "f_b",
        "c_s",
        "Rs",
        "rho0ang",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "f_b": 1.0e-5,
        "c_s": 0.02,
        "Rs": 0.02,
        "rho0ang": 0.0,
        "center_x": -10000.0,
        "center_y": -10000.0,
    }
    upper_limit_default = {
        "f_b": 1.0,
        "c_s": 1000.0,
        "Rs": 1000.0,
        "rho0ang": 1.0e25,
        "center_x": 10000.0,
        "center_y": 10000.0,
    }

    def __init__(
        self,
        r_min: float = 5e-5,
        r_max_factor: float = 10.0,
        num_bins: int = 400,
        **kwargs_numerics,
    ):
        """Initialization of the GreenBoschNFW class object.

        :param r_min: Minimum 2D radius of integration from subhalo center [arcseconds]
        :type r_min: Float
        :param r_max_factor: Maximum 2D radius of integration from subhalo center in
            units of scale radius [arcseconds/Rs]
        :type r_max_factor: Float
        :param num_bins: Number of log-spaced radial bins to integrate
        :type num_bins: Integer
        """

        self._last_params = None
        self._cached = None
        super().__init__()
        self._rad_interp = RadialInterpolate(**kwargs_numerics)
        self.r_min = r_min
        self.r_max_factor = r_max_factor
        self.num_bins = num_bins

    def function(self, x, y, f_b, c_s, Rs, rho0ang, center_x, center_y):
        """Lensing potential of the GreenBoschNFW profile.

        :param x: Angular position [arcseconds]
        :type x: Float
        :param y: Angular position [arcseconds]
        :type y: Float
        :param f_b: Instantaneous bound mass fraction relative to infall mass (M_bound /
            M_infall)
        :type f_b: Float
        :param c_s: Infall NFW concentration (R_virial / R_scale)
        :type c_s: Float
        :param Rs: Infall NFW scale radius [arcseconds]
        :type Rs: Float
        :param rho0ang: Dimensionless NFW normalization
        :type rho0ang: Float
        :param center_x: Position of halo center [arcseconds]
        :type center_x: Float
        :param center_y: Position of halo center [arcseconds]
        :type center_y: Float
        :return: Lensing potential enclosing radius r
        :rtype: Float
        """

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0ang)

        return self._rad_interp.function(
            x, y, r_bin=r_bin, kappa_r=kappa_r, center_x=center_x, center_y=center_y
        )

    def derivatives(self, x, y, f_b, c_s, Rs, rho0ang, center_x, center_y):
        """Returns first derivatives of the lensing potential, df/dx and df/dy.

        :param x: Angular position [arcseconds]
        :type x: Float
        :param y: Angular position [arcseconds]
        :type y: Float
        :param f_b: Instantaneous bound mass fraction relative to infall mass (M_bound /
            M_infall)
        :type f_b: Float
        :param c_s: Infall NFW concentration (R_virial / R_scale)
        :type c_s: Float
        :param Rs: Infall NFW scale radius [arcseconds]
        :type Rs: Float
        :param rho0ang: Dimensionless NFW normalization
        :type rho0ang: Float
        :param center_x: Position of halo center [arcseconds]
        :type center_x: Float
        :param center_y: Position of halo center [arcseconds]
        :type center_y: Float
        :return: f_x, f_y at interpolated positions (x, y)
        """

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0ang)

        return self._rad_interp.derivatives(
            x, y, r_bin=r_bin, kappa_r=kappa_r, center_x=center_x, center_y=center_y
        )

    def hessian(self, x, y, f_b, c_s, Rs, rho0ang, center_x, center_y):
        """Returns Hessian matrix/second derivates of the lensing potential, d^2f/dx^2,
        d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: Angular position [arcseconds]
        :type x: Float
        :param y: Angular position [arcseconds]
        :type y: Float
        :param f_b: Instantaneous bound mass fraction relative to infall mass (M_bound /
            M_infall)
        :type f_b: Float
        :param c_s: Infall NFW concentration (R_virial / R_scale)
        :type c_s: Float
        :param Rs: Infall NFW scale radius [arcseconds]
        :type Rs: Float
        :param rho0ang: Dimensionless NFW normalization
        :type rho0ang: Float
        :param center_x: Position of halo center [arcseconds]
        :type center_x: Float
        :param center_y: Position of halo center [arcseconds]
        :type center_y: Float
        :return: f_xx, f_xy, f_yx, f_yy at interpolated positions (x, y)
        """

        kappa_r, r_bin = self.rbin_kappa_r(f_b, c_s, Rs, rho0ang)

        return self._rad_interp.hessian(
            x, y, r_bin=r_bin, kappa_r=kappa_r, center_x=center_x, center_y=center_y
        )

    def set_dynamic(self):
        """

        :return: no return, deletes the pre-computed \\kappa(r) and rbin, for every instance of this class (subhalo)
        """

        self._last_params = None
        self._cached = None
        self._rad_interp.set_dynamic()

    def rho_3d_lens(self, r, f_b, c_s, Rs, rho0ang):
        """Returns the 3D density profile of the subhalo.

        :param r: 3D radius from the halo center
        :type r: Float
        :param f_b: Instantaneous bound mass fraction relative to infall mass (M_bound /
            M_infall)
        :type f_b: Float
        :param c_s: Infall NFW concentration (R_virial / R_scale)
        :type c_s: Float
        :param Rs: Infall NFW scale radius [arcseconds]
        :type Rs: Float
        :param rho0ang: Dimensionless NFW normalization
        :type rho0ang: Float
        :param center_x: Position of halo center [arcseconds]
        :return: Density \\rho(r)
        """

        a1, a2, a3, a4 = 0.338, 0.0, 0.157, 1.337
        b1, b2, b3, b4, b5, b6 = 0.448, 0.272, -0.199, 0.011, -1.119, 0.093
        c0, c1, c2, c3, c4 = 2.779, -0.035, -0.337, -0.099, 0.415
        f_te = f_b ** (a1 * (c_s / 10.0) ** a2) * c_s ** (a3 * (1.0 - f_b) ** a4)
        r_te = (
            c_s
            * f_b ** (b1 * (c_s / 10.0) ** b2)
            * c_s ** (b3 * (1.0 - f_b) ** b4)
            * np.exp(b5 * (c_s / 10.0) ** b6 * (1.0 - f_b))
        )
        delta = c0 * f_b ** (c1 * (c_s / 10.0) ** c2) * c_s ** (c3 * (1.0 - f_b) ** c4)
        coeff = (c_s - r_te) / (c_s * r_te)

        return (f_te * rho0ang) / (
            (1 + (coeff * r / Rs) ** delta) * (r / Rs) * (1 + r / Rs) ** 2
        )

    def rbin_kappa_r(self, f_b, c_s, Rs, rho0ang):
        """Returns the radial bins and the 2D radial convergence kappa(r), where r is in
        arcseconds.

        :param f_b: Instantaneous bound mass fraction relative to infall mass (M_bound /
            M_infall)
        :type f_b: Float
        :param c_s: Infall NFW concentration (R_virial / R_scale)
        :type c_s: Float
        :param Rs: Infall NFW scale radius [arcseconds]
        :type Rs: Float
        :param rho0ang: Dimensionless NFW normalization
        :type rho0ang: Float
        :param center_x: Position of halo center [arcseconds]
        :return: Radial convergence \\kappa(r)
        """

        def _round_params(p):
            return tuple(np.round(p, decimals=10))

        params = _round_params((f_b, c_s, Rs, rho0ang))

        if self._last_params == params and self._cached is not None:
            return self._cached

        r_bin = np.logspace(
            np.log10(self.r_min), np.log10(self.r_max_factor * Rs), self.num_bins
        )
        kappa_vals = []

        for r in r_bin:

            integrand = lambda z: 2.0 * self.rho_3d_lens(
                np.hypot(r, z), f_b, c_s, Rs, rho0ang
            )
            kappa, error = quad(
                integrand, 0, np.inf, limit=800, epsrel=1e-10, epsabs=1e-12
            )

            kappa_vals.append(kappa)

        kappa_r = np.array(kappa_vals)
        self._last_params = params
        self._cached = (kappa_r, r_bin)

        return kappa_r, r_bin

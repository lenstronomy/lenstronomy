import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from scipy.interpolate import interp1d

__all__ = ["RadialInterpolate"]


class RadialInterpolate(LensProfileBase):
    """
    radially interpolated profile with azimuthal symmetry
    """

    param_names = [
        "r_bin",
        "kappa_r",
    ]
    lower_limit_default = {}
    upper_limit_default = {}

    def function(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
    ):
        """Lensing potential (only needed for specific calculations, such as time
        delays)

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: lensing potential
        """
        return 0

    def derivatives(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
    ):
        """Returns df/dx and df/dy of the function.

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: f_x, f_y at interpolated positions (x, y)
        """
        return 0

    def hessian(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
    ):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: f_xx, f_xy, f_yx, f_yy at interpolated positions (x, y)
        """
        return 0

    def _kappa_r_interp(self, r, r_bin, kappa_r):
        """

        :param r:
        :param r_bin:
        :param kappa_r:
        :return:
        """
        if not hasattr(self, '_interp_kappa'):
            self._interp_kappa = interp1d(r_bin, kappa_r, fill_value=(kappa_r[0], kappa_r[-1]))
        return self._interp_kappa(r)

    def _mass_enclosed(self, r, r_bin, kappa_r):
        """convergence enclosed a radius

        :param r: radius
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: integrated convergence within radius <r
        """
        def _integrand(r):
            return r * 2 * np.py * self._kappa_r_interp(r, r_bin, kappa_r)

        from scipy import integrate

        # using scipy.integrate.quad() method
        geek = integrate.quad(_integrand, 0, r)
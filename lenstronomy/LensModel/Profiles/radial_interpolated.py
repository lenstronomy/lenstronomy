import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import param_util
from scipy.interpolate import interp1d
from scipy import integrate

__all__ = ["RadialInterpolate"]


class RadialInterpolate(LensProfileBase):
    """Radially interpolated profile with azimuthal symmetry."""

    param_names = [
        "r_bin",
        "kappa_r",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {}
    upper_limit_default = {}

    def function(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
        center_x=0,
        center_y=0,
    ):
        """Lensing potential (only needed for specific calculations, such as time
        delays)

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :param center_x: x-position of center of radial density profile
        :type center_x: float
        :param center_y: y-position of center of radial density profile
        :type center_y: float
        :return: lensing potential
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        # -\int alpha(r) dr  from 0 to r
        pot = self._potential_r(r, r_bin, kappa_r)
        return pot

    def derivatives(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
        center_x=0,
        center_y=0,
    ):
        """Returns df/dx and df/dy of the function.

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :param center_x: x-position of center of radial density profile
        :type center_x: float
        :param center_y: y-position of center of radial density profile
        :type center_y: float
        :return: f_x, f_y at interpolated positions (x, y)
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.maximum(np.sqrt(x_**2 + y_**2), 10 ** (-10))
        alpha = self.alpha(r, r_bin, kappa_r)
        return alpha * x_ / r, alpha * y_ / r

    def hessian(
        self,
        x,
        y,
        r_bin=None,
        kappa_r=None,
        center_x=0,
        center_y=0,
    ):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2.

        :param x: x-coordinate (angular position), float or numpy array
        :param y: y-coordinate (angular position), float or numpy array
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :param center_x: x-position of center of radial density profile
        :type center_x: float
        :param center_y: y-position of center of radial density profile
        :type center_y: float
        :return: f_xx, f_xy, f_yx, f_yy at interpolated positions (x, y)
        """
        r, phi = param_util.cart2polar(x, y, center_x=center_x, center_y=center_y)
        r = np.maximum(r, 10 ** (-10))
        kappa = self._kappa_r_interp(r, r_bin, kappa_r)
        # shear in the spherical case is the average convergence enclosed minus the convergence at the radius
        # source: Kaiser 1995
        gamma = 1 * (self._mass_enclosed(r, r_bin, kappa) / (np.pi * r**2) - kappa)
        # turn in to vector for gamma and cartesian derivatives
        gamma1 = -np.cos(2.0 * phi) * gamma
        gamma2 = -np.sin(2.0 * phi) * gamma
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return f_xx, f_xy, f_xy, f_yy

    def alpha(self, r, r_bin, kappa_r):
        """Radial deflection angle m(<r) / r / pi.

        :param r: radius from center
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: radial deflection angle
        """
        r_ = np.maximum(r, 10 ** (-10))
        return self._mass_enclosed(r, r_bin, kappa_r) / r_ / np.pi

    def _kappa_r_interp(self, r, r_bin, kappa_r):
        """Calls interpolated kappa(r)

        :param r: radius
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: numpy array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: kappa(r)
        """
        if not hasattr(self, "_interp_kappa"):
            self._interp_kappa = interp1d(
                r_bin, kappa_r, fill_value=(kappa_r[0], kappa_r[-1]), bounds_error=False
            )
        return self._interp_kappa(r)

    def _mass_enclosed(self, r, r_bin, kappa_r):
        """Convergence enclosed a radius.

        :param r: radius
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: numpy array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: integrated convergence within radius <r
        """
        if not hasattr(self, "_interp_m_enclosed"):

            def _integrand(x):
                return x * 2 * np.pi * self._kappa_r_interp(x, r_bin, kappa_r)

            m_slice_list = []
            r_min = 0
            for r_ in r_bin:
                m_slice, _ = integrate.quad(_integrand, r_min, r_)
                m_slice_list.append(m_slice)
                r_min = r_
            m_r = np.cumsum(m_slice_list)
            self._interp_m_enclosed = interp1d(
                r_bin, m_r, fill_value=(0, m_r[-1]), bounds_error=False
            )
        return self._interp_m_enclosed(r)

    def _potential_r(self, r, r_bin, kappa_r):
        """Convergence enclosed a radius.

        :param r: radius
        :param r_bin: radial bins for which convergence values are provided
        :type r_bin: numpy array
        :param kappa_r: convergence values corresponding to the r_bin radii
        :type kappa_r: array of same size as r_bin
        :return: integrated convergence within radius <r
        """
        if not hasattr(self, "_interp_potential"):

            def _integrand(x):
                return self.alpha(x, r_bin, kappa_r)

            pot_slice_list = []
            r_min = 0
            for r_ in r_bin:
                pot_slice, _ = integrate.quad(_integrand, r_min, r_)
                pot_slice_list.append(pot_slice)
                r_min = r_
            pot_r = np.cumsum(pot_slice_list)
            self._interp_potential = interp1d(
                r_bin, pot_r, fill_value=(0, pot_r[-1]), bounds_error=False
            )
        return self._interp_potential(r)

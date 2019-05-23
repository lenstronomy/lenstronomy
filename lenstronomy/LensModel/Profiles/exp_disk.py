__author__ = 'dgilman'

from scipy.integrate import quad
import numpy as np
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.Util import param_util

class EXPDISK(object):
    """
    this class contains the function and the derivatives of an elliptical exponential profile
    with the ellipticity included in the convergence (not the potential)
    """
    param_names = ['k_eff', 'R_eff', 'n', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'k_eff': 0, 'R_eff': 0, 'n': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'k_eff': 10, 'R_eff': 100, 'n': 8, 'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def __init__(self):

        self._sersic = Sersic()

    def function(self, x, y, n, R_eff, k_eff, e1, e2, center_x=0, center_y=0):

        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        integral = quad(self._integrand_I, 0, 1, args=(x_, y_, q, n, R_eff, k_eff, center_x, center_y))[0]

        return 0.5 * q * integral

    def _compute_derivative_atcoord(self, x, y, n, R_eff, k_eff, e1, e2, center_x=0, center_y = 0):

        phi_G, q = param_util.ellipticity2phi_gamma(e1, e2)
        x_, y_ = self._coord_rotate(x, y, phi_G, center_x, center_y)
        y *= q ** -1
        alpha_x = x_ * q * quad(self._integrand_J, 0, 1, args=(x_, y_, n, q, R_eff, k_eff, 0))[0]
        alpha_y = y_ * q * quad(self._integrand_J, 0, 1, args=(x_, y_, n, q, R_eff, k_eff, 1))[0]
        alpha_x, alpha_y = self._coord_rotate(alpha_x, alpha_y, -phi_G, 0, 0)
        return alpha_x, alpha_y

    def derivatives(self, x, y, n, R_eff, k_eff, e1, e2, center_x=0, center_y = 0):

        if isinstance(x, float) and isinstance(y, float):

            alpha_x, alpha_y = self._compute_derivative_atcoord(x, y, n, R_eff, k_eff,
                                        e1, e2, center_x=center_x, center_y = center_y)

        else:
            assert isinstance(x, np.ndarray) or isinstance(x, list)
            assert isinstance(y, np.ndarray) or isinstance(y, list)
            x = np.array(x)
            y = np.array(y)
            shape0 = x.shape
            assert shape0 == y.shape

            alpha_x, alpha_y = np.empty_like(x).ravel(), np.empty_like(y).ravel()

            for i, (x_i, y_i) in enumerate(zip(x.ravel(), y.ravel())):

                fxi, fyi = self._compute_derivative_atcoord(x_i, y_i, n, R_eff, k_eff,
                                   e1, e2, center_x=center_x, center_y = center_y)

                alpha_x[i], alpha_y[i] = fxi, fyi

            alpha_x.reshape(shape0)
            alpha_y.reshape(shape0)

        return alpha_x, alpha_y

    def _kappa(self, R_ellipse, n, R_eff, k_eff):

        exponent = - (R_ellipse * R_eff**-1) ** (-n**-1)

        return k_eff * np.exp(exponent)

    def _integrand_J(self, u, x, y, n, q, R_eff, k_eff, n_integral):

        R_ellipse = self._elliptical_coord_u(x, y, u, q) ** 2
        kappa = self._kappa(R_ellipse, n, R_eff, k_eff)

        return kappa * (1 - (1 - q**2)*u) ** (n_integral + 0.5)

    def _elliptical_coord_u(self, x, y, u, q):

        return (u * (x**2 + y**2 * (x**2 + (1-q**2)*u)**-1))**0.5

    def _integrand_I(self, u, x, y, q, n, reff, keff, centerx, centery):

        ellip_coord = self._elliptical_coord_u(x, y, u, q)

        def_angle_circular = self._sersic.alpha_abs(ellip_coord, 0, n, reff, keff, centerx, centery)

        return ellip_coord * def_angle_circular * (1 - (1-q**2)*u) ** -0.5 * u ** -1

    def _coord_rotate(self, x, y, phi_G, center_x, center_y):

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        return x_, y_

    def _coord_transf(self, x, y, q, phi_G, center_x, center_y):
        """

        :param x:
        :param y:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
        return x_, y_


expdisk = EXPDISK()

xvals = np.linspace(0.1, 10, 100)

n = 4
reff = 2
keff = 1
e1 = 0.4
e2 = 0.

fx, fy = expdisk.derivatives(xvals, np.zeros_like(xvals), n, reff, keff, e1, e2)

import matplotlib.pyplot as plt
plt.plot(xvals, fx)
plt.show()
__author__ = 'dgilman'

from scipy.integrate import quad
import numpy as np
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.Util import param_util

class SersicEllipseKappa(object):
    """
    this class contains the function and the derivatives of an elliptical sersic profile
    with the ellipticity introduced in the convergence (not the potential).

    This requires the use of numerical integrals (Keeton 2004)
    """
    param_names = ['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'k_eff': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'k_eff': 10, 'R_sersic': 100, 'n_sersic': 8, 'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def __init__(self):

        self._sersic = Sersic()

    def function(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y=0):

        raise Exception('not yet implemented')

        # phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        #
        # if isinstance(x, float) and isinstance(y, float):
        #
        #     x_, y_ = self._coord_rotate(x, y, phi_G, center_x, center_y)
        #     integral = quad(self._integrand_I, 0, 1, args=(x_, y_, q, n_sersic, R_sersic, k_eff, center_x, center_y))[0]
        #
        # else:
        #
        #     assert isinstance(x, np.ndarray) or isinstance(x, list)
        #     assert isinstance(y, np.ndarray) or isinstance(y, list)
        #     x = np.array(x)
        #     y = np.array(y)
        #     shape0 = x.shape
        #     assert shape0 == y.shape
        #
        #     if isinstance(phi_G, float) or isinstance(phi_G, int):
        #         phiG = np.ones_like(x) * float(phi_G)
        #         q = np.ones_like(x) * float(q)
        #     integral = []
        #     for i, (x_i, y_i, phi_i, q_i) in \
        #             enumerate(zip(x.ravel(), y.ravel(), phiG.ravel(), q.ravel())):
        #
        #         integral.append(quad(self._integrand_I, 0, 1, args=(x_, y_, q, n_sersic,
        #                                                             R_sersic, k_eff, center_x, center_y))[0])
        #
        #
        # return 0.5 * q * integral

    def derivatives(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y = 0):

        phi_G, gam = param_util.ellipticity2phi_gamma(e1, e2)
        q = max(1-gam, 0.00001)

        x, y = self._coord_rotate(x, y, phi_G, center_x, center_y)

        if isinstance(x, float) and isinstance(y, float):

            alpha_x, alpha_y = self._compute_derivative_atcoord(x, y, n_sersic, R_sersic, k_eff,
                                        phi_G, q, center_x=center_x, center_y = center_y)

        else:

            assert isinstance(x, np.ndarray) or isinstance(x, list)
            assert isinstance(y, np.ndarray) or isinstance(y, list)
            x = np.array(x)
            y = np.array(y)
            shape0 = x.shape
            assert shape0 == y.shape

            alpha_x, alpha_y = np.empty_like(x).ravel(), np.empty_like(y).ravel()

            if isinstance(phi_G, float) or isinstance(phi_G, int):
                phiG = np.ones_like(alpha_x) * float(phi_G)
                q = np.ones_like(alpha_x) * float(q)

            for i, (x_i, y_i, phi_i, q_i) in \
                    enumerate(zip(x.ravel(), y.ravel(), phiG.ravel(), q.ravel())):

                fxi, fyi = self._compute_derivative_atcoord(x_i, y_i, n_sersic, R_sersic, k_eff,
                                   phi_i, q_i, center_x=center_x, center_y = center_y)

                alpha_x[i], alpha_y[i] = fxi, fyi

            alpha_x = alpha_x.reshape(shape0)
            alpha_y = alpha_y.reshape(shape0)

        alpha_x, alpha_y = self._coord_rotate(alpha_x, alpha_y, -phi_G, 0, 0)

        return alpha_x, alpha_y

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x=0, center_y = 0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, n_sersic, R_sersic, k_eff, e1, e2, center_x, center_y)
        diff = 0.000001
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, n_sersic, R_sersic, k_eff, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, n_sersic, R_sersic, k_eff, e1, e2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        #f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff

        return f_xx, f_yy, f_xy

    def projected_mass(self, x, y, q, n_sersic, R_sersic, k_eff, u = 1, power = 1):

        b_n = self._sersic.b_n(n_sersic)

        elliptical_coord = self._elliptical_coord_u(x, y, u, q) ** power
        elliptical_coord *= R_sersic ** -power

        exponent = -b_n * (elliptical_coord**(1./n_sersic) - 1)

        return k_eff * np.exp(exponent)

    def _integrand_J(self, u, x, y, n_sersic, q, R_sersic, k_eff, n_integral):

        kappa = self.projected_mass(x, y, q, n_sersic, R_sersic, k_eff, u = u, power=1)

        power = -(n_integral + 0.5)

        return kappa * (1 - (1 - q**2)*u) ** power

    def _integrand_I(self, u, x, y, q, n_sersic, R_sersic, keff, centerx, centery):

        ellip_coord = self._elliptical_coord_u(x, y, u, q)

        def_angle_circular = self._sersic.alpha_abs(ellip_coord, 0, n_sersic, R_sersic, keff, centerx, centery)

        return ellip_coord * def_angle_circular * (1 - (1-q**2)*u) ** -0.5 * u ** -1

    def _compute_derivative_atcoord(self, x, y, n_sersic, R_sersic, k_eff, phi_G, q, center_x=0, center_y = 0):

        alpha_x = x * q * quad(self._integrand_J, 0, 1, args=(x, y, n_sersic, q, R_sersic, k_eff, 0))[0]
        alpha_y = y * q * quad(self._integrand_J, 0, 1, args=(x, y, n_sersic, q, R_sersic, k_eff, 1))[0]

        return alpha_x, alpha_y

    @staticmethod
    def _elliptical_coord_u(x, y, u, q):

        fac = 1 - (1 - q**2) * u

        return (u * (x**2 + y**2 * fac**-1) )**0.5

    @staticmethod
    def _coord_rotate(x, y, phi_G, center_x, center_y):

        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x_ = cos_phi * x_shift + sin_phi * y_shift
        y_ = -sin_phi * x_shift + cos_phi * y_shift

        return x_, y_

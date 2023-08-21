from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSEMajorAxisSet
import numpy as np

__all__ = ['HernquistEllipseCSE']


class HernquistEllipseCSE(Hernquist_Ellipse):
    """
    this class contains functions for the elliptical Hernquist profile. Ellipticity is defined in the convergence.
    Approximation with CSE profile introduced by Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf


    """
    param_names = ['sigma0', 'Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'sigma0': 0, 'Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'sigma0': 100, 'Rs': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.cse_major_axis_set = CSEMajorAxisSet()
        # Table 2 in Oguri 2021
        self._a_list = [9.200445e-18, 2.184724e-16, 3.548079e-15, 2.823716e-14, 1.091876e-13,
                        6.998697e-13, 3.142264e-12, 1.457280e-11, 4.472783e-11, 2.042079e-10,
                        8.708137e-10, 2.423649e-09, 7.353440e-09, 5.470738e-08, 2.445878e-07,
                        4.541672e-07, 3.227611e-06, 1.110690e-05, 3.725101e-05, 1.056271e-04,
                        6.531501e-04, 2.121330e-03, 8.285518e-03, 4.084190e-02, 5.760942e-02,
                        1.788945e-01, 2.092774e-01, 3.697750e-01, 3.440555e-01, 5.792737e-01,
                        2.325935e-01, 5.227961e-01, 3.079968e-01, 1.633456e-01, 7.410900e-02,
                        3.123329e-02, 1.292488e-02, 2.156527e+00, 1.652553e-02, 2.314934e-02,
                        3.992313e-01]
        self._s_list = [1.199110e-06, 3.751762e-06, 9.927207e-06, 2.206076e-05, 3.781528e-05,
                        6.659808e-05, 1.154366e-04, 1.924150e-04, 3.040440e-04, 4.683051e-04,
                        7.745084e-04, 1.175953e-03, 1.675459e-03, 2.801948e-03, 9.712807e-03,
                        5.469589e-03, 1.104654e-02, 1.893893e-02, 2.792864e-02, 4.152834e-02,
                        6.640398e-02, 1.107083e-01, 1.648028e-01, 2.839601e-01, 4.129439e-01,
                        8.239115e-01, 6.031726e-01, 1.145604e+00, 1.401895e+00, 2.512223e+00,
                        2.038025e+00, 4.644014e+00, 9.301590e+00, 2.039273e+01, 4.896534e+01,
                        1.252311e+02, 3.576766e+02, 2.579464e+04, 2.944679e+04, 2.834717e+03,
                        5.931328e+04]
        super(HernquistEllipseCSE, self).__init__()

    def function(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns double integral of NFW profile
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        f_ = self.cse_major_axis_set.function(x__ / Rs, y__ / Rs, self._a_list, self._s_list, q)
        const = self._normalization(sigma0, Rs, q)
        return const * f_

    def derivatives(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)
        f__x, f__y = self.cse_major_axis_set.derivatives(x__ / Rs, y__ / Rs, self._a_list, self._s_list, q)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        const = self._normalization(sigma0, Rs, q) / Rs
        return const * f_x, const * f_y

    def hessian(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)
        f__xx, f__xy, __, f__yy = self.cse_major_axis_set.hessian(x__ / Rs, y__ / Rs, self._a_list, self._s_list, q)

        # rotate back
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_q) * gamma1__ - np.sin(2 * phi_q) * gamma2__
        gamma2 = +np.sin(2 * phi_q) * gamma1__ + np.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        const = self._normalization(sigma0, Rs, q) / Rs ** 2

        return const * f_xx, const * f_xy, const * f_xy, const * f_yy

    @staticmethod
    def _normalization(sigma0, Rs, q):
        """
        mapping to eqn 10 and 11 in Oguri 2021 from phenomenological definition

        :param sigma0: sigma0 normalization
        :param Rs: scale radius
        :param q: axis ratio
        :return: normalization (m)
        """
        rs_ = Rs / np.sqrt(q)
        const = sigma0 / 2 * rs_ ** 3
        return const

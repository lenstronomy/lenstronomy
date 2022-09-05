__author__ = 'sibirrer'

import numpy as np
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSEProductAvgSet
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSEMajorAxisSet
import lenstronomy.Util.param_util as param_util

__all__ = ['NFW_ELLIPSE_CSE']


class NFW_ELLIPSE_CSE(NFW_ELLIPSE):
    """
    this class contains functions concerning the NFW profile with an ellipticity defined in the convergence
    parameterization of alpha_Rs and Rs is the same as for the spherical NFW profile
    Approximation with CSE profile introduced by Oguri 2021: https://arxiv.org/pdf/2106.11464.pdf
    Match to NFW using CSEs is approximate: kappa matches to ~1-2%

    relation are: R_200 = c * Rs


    """
    profile_name = 'NFW_ELLIPSE_CSE'
    param_names = ['Rs', 'alpha_Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, high_accuracy=True):
        """

        :param high_accuracy: boolean, if True uses a more accurate larger set of CSE profiles (see Oguri 2021)
        """
        self.cse_major_axis_set = CSEProductAvgSet()
        self.nfw = NFW()
        if high_accuracy is True:
            # Table 1 in Oguri 2021
            self._s_list = [1.082411e-06, 8.786566e-06, 3.292868e-06, 1.860019e-05, 3.274231e-05,
                            6.232485e-05, 9.256333e-05, 1.546762e-04, 2.097321e-04, 3.391140e-04,
                            5.178790e-04, 8.636736e-04, 1.405152e-03, 2.193855e-03, 3.179572e-03,
                            4.970987e-03, 7.631970e-03, 1.119413e-02, 1.827267e-02, 2.945251e-02,
                            4.562723e-02, 6.782509e-02, 1.596987e-01, 1.127751e-01, 2.169469e-01,
                            3.423835e-01, 5.194527e-01, 8.623185e-01, 1.382737e+00, 2.034929e+00,
                            3.402979e+00, 5.594276e+00, 8.052345e+00, 1.349045e+01, 2.603825e+01,
                            4.736823e+01, 6.559320e+01, 1.087932e+02, 1.477673e+02, 2.495341e+02,
                            4.305999e+02, 7.760206e+02, 2.143057e+03, 1.935749e+03]
            self._a_list = [1.648988e-18, 6.274458e-16, 3.646620e-17, 3.459206e-15, 2.457389e-14,
                            1.059319e-13, 4.211597e-13, 1.142832e-12, 4.391215e-12, 1.556500e-11,
                            6.951271e-11, 3.147466e-10, 1.379109e-09, 3.829778e-09, 1.384858e-08,
                            5.370951e-08, 1.804384e-07, 5.788608e-07, 3.205256e-06, 1.102422e-05,
                            4.093971e-05, 1.282206e-04, 4.575541e-04, 7.995270e-04, 5.013701e-03,
                            1.403508e-02, 5.230727e-02, 1.898907e-01, 3.643448e-01, 7.203734e-01,
                            1.717667e+00, 2.217566e+00, 3.187447e+00, 8.194898e+00, 1.765210e+01,
                            1.974319e+01, 2.783688e+01, 4.482311e+01, 5.598897e+01, 1.426485e+02,
                            2.279833e+02, 5.401335e+02, 9.743682e+02, 1.775124e+03]

        else:
            # Table 3 in Oguri 2021
            self._a_list = [1.434960e-16, 5.232413e-14, 2.666660e-12, 7.961761e-11, 2.306895e-09,
                            6.742968e-08, 1.991691e-06, 5.904388e-05, 1.693069e-03, 4.039850e-02,
                            5.665072e-01, 3.683242e+00, 1.582481e+01, 6.340984e+01, 2.576763e+02,
                            1.422619e+03]
            self._s_list = [4.041628e-06, 3.086267e-05, 1.298542e-04, 4.131977e-04, 1.271373e-03,
                            3.912641e-03, 1.208331e-02, 3.740521e-02, 1.153247e-01, 3.472038e-01,
                            1.017550e+00, 3.253031e+00, 1.190315e+01, 4.627701e+01, 1.842613e+02,
                            8.206569e+02]

        super(NFW_ELLIPSE_CSE, self).__init__()

    def function(self, x, y, Rs, alpha_Rs, e1, e2, center_x=0, center_y=0):
        """
        returns elliptically distorted NFW lensing potential

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: lensing potential
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)

        # potential calculation
        f_ = self.cse_major_axis_set.function(x__/Rs, y__/Rs, self._a_list, self._s_list, q)
        const = self._normalization(alpha_Rs, Rs, q)
        return const * f_

    def derivatives(self, x, y, Rs, alpha_Rs, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function, calculated as an elliptically distorted deflection angle of the
        spherical NFW profile

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: deflection in x-direction, deflection in y-direction
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)
        f__x, f__y = self.cse_major_axis_set.derivatives(x__/Rs, y__/Rs, self._a_list, self._s_list, q)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        const = self._normalization(alpha_Rs, Rs, q) / Rs
        return const * f_x, const * f_y

    def hessian(self, x, y, Rs, alpha_Rs, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        the calculation is performed as a numerical differential from the deflection field.
        Analytical relations are possible.

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param center_x: center of halo (in angular units)
        :param center_y: center of halo (in angular units)
        :return: d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        phi_q, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_q)
        f__xx, f__xy, __, f__yy = self.cse_major_axis_set.hessian(x__/Rs, y__/Rs, self._a_list, self._s_list, q)

        # rotate back
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_q) * gamma1__ - np.sin(2 * phi_q) * gamma2__
        gamma2 = +np.sin(2 * phi_q) * gamma1__ + np.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        const = self._normalization(alpha_Rs, Rs, q) / Rs**2

        return const * f_xx, const * f_xy, const * f_xy, const * f_yy

    def _normalization(self, alpha_Rs, Rs, q):
        """
        applying to eqn 7 and 8 in Oguri 2021 from phenomenological definition

        :param alpha_Rs: deflection at Rs
        :param Rs: scale radius
        :param q: axis ratio
        :return: normalization (m)
        """
        rho0 = self.nfw.alpha2rho0(alpha_Rs, Rs)
        rs_ = Rs
        const = 4 * rho0 * rs_ ** 3
        return const

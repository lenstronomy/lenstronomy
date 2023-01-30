__author__ = 'mgomer'

import numpy as np
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.general_nfw import GNFW
# from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
from lenstronomy.LensModel.Profiles.cored_steep_ellipsoid import CSEProductAvg, CSEProductAvgSet
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel

__all__ = ['GNFW_ELLIPSE_CSE']


class GNFW_ELLIPSE_CSE(GNFW):
    """
    This class is used to approximate a generalized NFW profile (see LensModel.Profiles.general_nfw) using
    a sum of CSE profiles (same method as lenstronomy.LensModel.Profiles.nfw_ellipse_cse).
    This enables an elliptical version of a GNFW profile.

    Approximated over the range from 10^(-6) * Rs to 10^(3) * Rs
    Accuracy of the approximation can be estimated  using estimate_deflection_error and differs depending
    on inner slope and outer slope. Accurate to ~3% or better for most values of inner slope < 2 and outer slope > 2

    """
    profile_name = 'GNFW_ELLIPSE_CSE'
    param_names = ['Rs', 'alpha_Rs', 'e1','e2', 'gamma_inner', 'gamma_outer','center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'e1': -0.5, 'e2': -0.5, 'gamma_inner': 0.1, 'gamma_outer': 1.0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'e1': 0.5, 'e2': 0.5, 'gamma_inner': 2.9, 'gamma_outer': 10.0, 'center_x': 100, 'center_y': 100}

    def __init__(self, num_components=70):
        self.num_components=num_components
        self.cse_major_axis_set = CSEProductAvgSet()
        self.s_list=np.logspace(-6,3,num_components)
        # self.a_list=self.linear_fit(self.s_list)
        self.gnfw=GNFW()
        self.cse_product_avg=CSEProductAvg()
        self.single_cse=LensModel(['CSE'])

    def linear_fit(self, gamma_inner, gamma_outer):
        """
        AX=Y, where A is a matrix describing all the components and X is the coefficients to be solved and Y is the target profile
        Inverts the matrix to find the set of amplitudes for each CSE to return the correct values at each r

        :return: vector of amplitudes for each CSE
        """
        self.r_eval_list=self.s_list #targets to evaluate must have same dimension as number of parameters to have an invertible matrix
        #Y is 1D kappa target function evaluated for all r values, setting Rs=1 and rho0=1
        Y=self.gnfw.density_2d(x=self.r_eval_list, y=np.zeros_like(self.r_eval_list), Rs=1, rho0=1, gamma_inner=gamma_inner, gamma_outer=gamma_outer)
        A_matrix = np.zeros((self.num_components, self.num_components))
        rs_ = 1
        const = 4 * 1 * rs_ ** 3 #normalization, see _normalization
        for j in range(self.num_components):
            #A[i,j] is the jth CSE 1D kappa evaluated at r[i].
            kwargs_lens=[{'a':1, 's':self.s_list[j], 'e1':0, 'e2':0, 'center_x':0, 'center_y':0}]
            A_matrix[:, j] = const * self.single_cse.kappa(x=self.r_eval_list, y=0, kwargs=kwargs_lens)

        return np.matmul(np.linalg.inv(A_matrix), Y)

    def estimate_deflection_error(self, gamma_inner, gamma_outer):
        """
        estimates the mean x-direction deflection error of the approximation over the radial range

        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
        :return: sqrt(mean(squared relative errors) over 100 radial bins
        """
        rtargets=np.logspace(-6,3,100)
        alpha_Rs = self.gnfw.rho02alpha(1, 1, gamma_inner, gamma_outer)
        cse_approx=np.array(self.derivatives(rtargets, np.zeros_like(rtargets), 1, alpha_Rs, 0, 0, gamma_inner, gamma_outer, center_x=0, center_y=0)[0])
        target=np.array(self.gnfw.derivatives(rtargets, np.zeros_like(rtargets), 1, alpha_Rs, gamma_inner, gamma_outer, center_x=0, center_y=0)[0])
        return np.sqrt(np.mean((cse_approx-target)**2/target**2)) #sqrt(mean(squared relative errors))

    def function(self, x, y, Rs, alpha_Rs, e1, e2, gamma_inner, gamma_outer, center_x=0, center_y=0):
        """
        returns elliptically distorted NFW lensing potential

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
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
        a_list = self.linear_fit(gamma_inner, gamma_outer)
        f_ = self.cse_major_axis_set.function(x__/Rs, y__/Rs, a_list, self.s_list, q)
        const = self._normalization(alpha_Rs, Rs, gamma_inner, gamma_outer)
        return const * f_

    def derivatives(self, x, y, Rs, alpha_Rs, e1, e2, gamma_inner, gamma_outer, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function, calculated as an elliptically distorted deflection angle of the
        spherical NFW profile

        :param x: angular position (normally in units of arc seconds)
        :param y: angular position (normally in units of arc seconds)
        :param Rs: turn over point in the slope of the NFW profile in angular unit
        :param alpha_Rs: deflection (angular units) at projected Rs
        :param e1: eccentricity component in x-direction
        :param e2: eccentricity component in y-direction
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
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
        a_list = self.linear_fit(gamma_inner, gamma_outer)
        f__x, f__y = self.cse_major_axis_set.derivatives(x__/Rs, y__/Rs, a_list, self.s_list, q)

        # rotate deflections back
        f_x, f_y = util.rotate(f__x, f__y, -phi_q)
        const = self._normalization(alpha_Rs, Rs, gamma_inner, gamma_outer) / Rs
        return const * f_x, const * f_y

    def hessian(self, x, y, Rs, alpha_Rs, e1, e2, gamma_inner, gamma_outer, center_x=0, center_y=0):
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
        :param gamma_inner: logarithmic profile slope interior to Rs
        :param gamma_outer: logarithmic profile slope outside Rs
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
        a_list = self.linear_fit(gamma_inner, gamma_outer)
        f__xx, f__xy, __, f__yy = self.cse_major_axis_set.hessian(x__/Rs, y__/Rs, a_list, self.s_list, q)

        # rotate back
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_q) * gamma1__ - np.sin(2 * phi_q) * gamma2__
        gamma2 = +np.sin(2 * phi_q) * gamma1__ + np.cos(2 * phi_q) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        const = self._normalization(alpha_Rs, Rs, gamma_inner, gamma_outer) / Rs**2

        return const * f_xx, const * f_xy, const * f_xy, const * f_yy

    def _normalization(self, alpha_Rs, Rs, gamma_inner, gamma_outer):
        """
        applying to eqn 7 and 8 in Oguri 2021 from phenomenological definition

        :param alpha_Rs: deflection at Rs
        :param Rs: scale radius
        :param q: axis ratio
        :return: normalization (m)
        """
        rho0 = self.gnfw.alpha2rho0(alpha_Rs, Rs, gamma_inner, gamma_outer)
        rs_ = Rs
        const = 4 * rho0 * rs_ ** 3
        return const


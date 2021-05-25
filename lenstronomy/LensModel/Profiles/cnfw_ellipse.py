__author__ = 'sibirrer'

#this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
#the potential therefore is its integral

import numpy as np
from lenstronomy.LensModel.Profiles.cnfw import CNFW
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['CNFW_ELLIPSE']


class CNFW_ELLIPSE(LensProfileBase):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    """
    param_names = ['Rs', 'alpha_Rs', 'r_core', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'r_core': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'r_core': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.cnfw = CNFW()
        self._diff = 0.0000000001
        super(CNFW_ELLIPSE, self).__init__()

    def function(self, x, y, Rs, alpha_Rs, r_core, e1, e2, center_x=0, center_y=0):
        """
        returns double integral of NFW profile
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = min(abs(1. - q), 0.99)
        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - e)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + e)
        R_ = np.sqrt(xt1**2 + xt2**2)
        f_ = self.cnfw.function(R_, 0, Rs, alpha_Rs, r_core, center_x=0, center_y=0)
        return f_

    def derivatives(self, x, y, Rs, alpha_Rs, r_core, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = min(abs(1. - q), 0.99)
        xt1 = (cos_phi*x_shift+sin_phi*y_shift)*np.sqrt(1 - e)
        xt2 = (-sin_phi*x_shift+cos_phi*y_shift)*np.sqrt(1 + e)

        f_x_prim, f_y_prim = self.cnfw.derivatives(xt1, xt2, Rs, alpha_Rs, r_core, center_x=0, center_y=0)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, r_core, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        diff = 0.0000001
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, Rs, alpha_Rs, r_core, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, Rs, alpha_Rs, r_core, e1, e2, center_x, center_y)

        alpha_ra_dx_, alpha_dec_dx_ = self.derivatives(x - diff, y, Rs, alpha_Rs, r_core, e1, e2, center_x, center_y)
        alpha_ra_dy_, alpha_dec_dy_ = self.derivatives(x, y - diff, Rs, alpha_Rs, r_core, e1, e2, center_x, center_y)

        dalpha_rara = (alpha_ra_dx - alpha_ra_dx_) / diff / 2
        dalpha_radec = (alpha_ra_dy - alpha_ra_dy_) / diff / 2
        dalpha_decra = (alpha_dec_dx - alpha_dec_dx_) / diff / 2
        dalpha_decdec = (alpha_dec_dy - alpha_dec_dy_) / diff / 2

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def mass_3d_lens(self, R, Rs, alpha_Rs, r_core, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r given a lens parameterization with angular units

        :return:
        """
        return self.cnfw.mass_3d_lens(R, Rs, alpha_Rs, r_core)

    def density_lens(self, R, Rs, alpha_Rs, r_core, e1=0, e2=0):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        """
        return self.cnfw.density_lens(R, Rs, alpha_Rs, r_core)

__author__ = 'sibirrer'

#this file contains a class to compute the Navaro-Frank-White function in mass/kappa space
#the potential therefore is its integral

import numpy as np
from lenstronomy.LensModel.Profiles.nfw import NFW
import lenstronomy.Util.param_util as param_util


class NFW_ELLIPSE(object):
    """
    this class contains functions concerning the NFW profile

    relation are: R_200 = c * Rs
    """
    param_names = ['Rs', 'theta_Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'theta_Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'theta_Rs': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpol=False, num_interp_X=1000, max_interp_X=10):
        self.nfw = NFW(interpol=interpol, num_interp_X=num_interp_X, max_interp_X=max_interp_X)
        self._diff = 0.0000000001

    def function(self, x, y, Rs, theta_Rs, e1, e2, center_x=0, center_y=0):
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
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        f_ = self.nfw.nfwPot(R_, Rs, rho0_input)
        return f_

    def derivatives(self, x, y, Rs, theta_Rs, e1, e2, center_x=0, center_y=0):
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
        R_ = np.sqrt(xt1**2 + xt2**2)
        rho0_input = self.nfw._alpha2rho0(theta_Rs=theta_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        f_x_prim, f_y_prim = self.nfw.nfwAlpha(R_, Rs, rho0_input, xt1, xt2)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, Rs, theta_Rs, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, Rs, theta_Rs, e1, e2, center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, Rs, theta_Rs, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, Rs, theta_Rs, e1, e2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff

        return f_xx, f_yy, f_xy

    def mass_3d_lens(self, R, Rs, theta_Rs, e1=1, e2=0):
        """

        :param R:
        :param Rs:
        :param theta_Rs:
        :param q:
        :param phi_G:
        :return:
        """
        return self.nfw.mass_3d(R, Rs, theta_Rs)

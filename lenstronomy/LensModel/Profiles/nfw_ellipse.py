__author__ = 'sibirrer'

import numpy as np
from lenstronomy.LensModel.Profiles.nfw import NFW
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['NFW_ELLIPSE']


class NFW_ELLIPSE(LensProfileBase):
    """
    this class contains functions concerning the NFW profile with an ellipticity defined in the potential
    parameterization of alpha_Rs and Rs is the same as for the spherical NFW profile

    from Glose & Kneib: https://cds.cern.ch/record/529584/files/0112138.pdf

    relation are: R_200 = c * Rs
    """
    profile_name = 'NFW_ELLIPSE'
    param_names = ['Rs', 'alpha_Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'Rs': 0, 'alpha_Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'Rs': 100, 'alpha_Rs': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, interpol=False, num_interp_X=1000, max_interp_X=10):
        """

        :param interpol: bool, if True, interpolates the functions F(), g() and h()
        :param num_interp_X: int (only considered if interpol=True), number of interpolation elements in units of r/r_s
        :param max_interp_X: float (only considered if interpol=True), maximum r/r_s value to be interpolated
         (returning zeros outside)
        """
        self.nfw = NFW(interpol=interpol, num_interp_X=num_interp_X, max_interp_X=max_interp_X)
        self._diff = 0.0000000001
        super(NFW_ELLIPSE, self).__init__()

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
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        R_ = np.sqrt(x_**2 + y_**2)
        rho0_input = self.nfw.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        f_ = self.nfw.nfwPot(R_, Rs, rho0_input)
        return f_

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
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        R_ = np.sqrt(x_ ** 2 + y_ ** 2)
        rho0_input = self.nfw.alpha2rho0(alpha_Rs=alpha_Rs, Rs=Rs)
        if Rs < 0.0000001:
            Rs = 0.0000001
        f_x_prim, f_y_prim = self.nfw.nfwAlpha(R_, Rs, rho0_input, x_, y_)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, Rs, alpha_Rs, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        the calculation is performed as a numerical differential from the deflection field. Analytical relations are possible

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
        alpha_ra, alpha_dec = self.derivatives(x, y, Rs, alpha_Rs, e1, e2, center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, Rs, alpha_Rs, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, Rs, alpha_Rs, e1, e2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff

        return f_xx, f_xy, f_yx, f_yy

    def mass_3d_lens(self, R, Rs, alpha_Rs, e1=1, e2=0):
        """

        :param R: radius (in angular units)
        :param Rs:
        :param alpha_Rs:
        :param e1:
        :param e2:
        :return:
        """
        return self.nfw.mass_3d_lens(R, Rs, alpha_Rs)

    def density_lens(self, r, Rs, alpha_Rs, e1=1, e2=0):
        """
        computes the density at 3d radius r given lens model parameterization.
        The integral in the LOS projection of this quantity results in the convergence quantity.

        :param r: 3d radios
        :param Rs: turn-over radius of NFW profile
        :param alpha_Rs: deflection at Rs
        :return: density rho(r)
        """
        return self.nfw.density_lens(r, Rs, alpha_Rs)

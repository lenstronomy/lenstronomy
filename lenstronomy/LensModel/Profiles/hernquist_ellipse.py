from lenstronomy.LensModel.Profiles.hernquist import Hernquist
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ['Hernquist_Ellipse']


class Hernquist_Ellipse(LensProfileBase):
    """
    this class contains functions for the elliptical Hernquist profile. Ellipticity is defined in the potential.


    """
    param_names = ['sigma0', 'Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'sigma0': 0, 'Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'sigma0': 100, 'Rs': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.spherical = Hernquist()
        self._diff = 0.0000000001
        super(Hernquist_Ellipse, self).__init__()

    def function(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns double integral of NFW profile
        """
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        f_ = self.spherical.function(x_, y_, sigma0, Rs)
        return f_

    def derivatives(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function (integral of NFW)
        """
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)

        f_x_prim, f_y_prim = self.spherical.derivatives(x_, y_, sigma0, Rs)
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi*f_x_prim-sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim+cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, sigma0, Rs, e1, e2, center_x, center_y)
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, sigma0, Rs, e1, e2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, sigma0, Rs, e1, e2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra)/diff
        f_xy = (alpha_ra_dy - alpha_ra)/diff
        f_yx = (alpha_dec_dx - alpha_dec)/diff
        f_yy = (alpha_dec_dy - alpha_dec)/diff
        return f_xx, f_xy, f_yx, f_yy

    def density(self, r, rho0, Rs, e1=0, e2=0):
        """
        computes the 3-d density

        :param r: 3-d radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: density at radius r
        """
        return self.spherical.density(r, rho0, Rs)

    def density_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """
        Density as a function of 3d radius in lensing parameters
        This function converts the lensing definition sigma0 into the 3d density

        :param r: 3d radius
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: enclosed mass in 3d
        """
        return self.spherical.density_lens(r, sigma0, Rs)

    def density_2d(self, x, y, rho0, Rs, e1=0, e2=0, center_x=0, center_y=0):
        """
        projected density along the line of sight at coordinate (x, y)

        :param x: x-coordinate
        :param y: y-coordinate
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: projected density
        """
        return self.spherical.density_2d(x, y, rho0, Rs, center_x, center_y)

    def mass_2d_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """
        mass enclosed projected 2d sphere of radius r
        Same as mass_2d but with input normalization in units of projected density
        :param r: projected radius
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: mass enclosed 2d projected radius
        """
        return self.spherical.mass_2d_lens(r, sigma0, Rs)

    def mass_2d(self, r, rho0, Rs, e1=0, e2=0):
        """
        mass enclosed projected 2d sphere of radius r

        :param r: projected radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: mass enclosed 2d projected radius
        """
        return self.spherical.mass_2d(r, rho0, Rs)

    def mass_3d(self, r, rho0, Rs, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r

        :param r: 3-d radius within the mass is integrated (same distance units as density definition)
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :return: enclosed mass
        """
        return self.spherical.mass_3d(r, rho0, Rs)

    def mass_3d_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """
        mass enclosed a 3d sphere or radius r in lensing parameterization

        :param r: 3-d radius within the mass is integrated (same distance units as density definition)
        :param sigma0: rho0 * Rs (units of projected density)
        :param Rs: Hernquist radius
        :return: enclosed mass
        """
        return self.spherical.mass_3d_lens(r, sigma0, Rs)

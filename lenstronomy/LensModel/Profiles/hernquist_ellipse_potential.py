__all__ = ["HernquistEllipsePotential"]

import numpy as np
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.hernquist import Hernquist
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase


class HernquistEllipsePotential(LensProfileBase):
    """This class implements the elliptical version of the Hernquist potential for
    gravitational lensing.

    The Hernquist profile, presented in Hernquist (1990),
    https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H/abstract, is a
    spherically symmetric density profile.

    This profile is defined by the density function:

    .. math::
        \\rho(R) = \\frac{\\rho_0}{\\left( \\frac{R}{R_s} \\right)
        \\left( 1 + \\frac{R}{R_s} \\right)^3}

    where :math:`\\rho_0` is the density normalization (`rho0`), and
    :math:`R_s` is the Hernquist radius (`Rs`). Here, we will use
    :math:`\\sigma_0 = \\rho_0 \\times R_s` as a parameter (`sigma0`) as it is
    more convenient to tune.

    In this implementation, the profile is generalized to include elliptical
    symmetry in the lensing potential rather than in the mass distribution. The
    potential ellipticity is parameterized by (`e1`, `e2`), and the profile is
    defined by :math:`\\sigma_0` (`sigma0`), :math:`R_s` (`Rs`), and a
    positional center (`center_x`, `center_y`).

    The ellipticity, :math:`e`, is defined as

    .. math::
        e = \\sqrt{e_1^2 + e_2^2} = \\equic \\frac{1 - q^2}{1 + q^2}

    where :math:`e_1` and :math:`e_2` are `e1` and `e2` respectively.
    """

    param_names = ["sigma0", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.spherical = Hernquist()
        self._diff = 0.00000001
        super(HernquistEllipsePotential, self).__init__()

    def function(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """Returns double integral of NFW profile.

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        # Maps (x, y) with (e1, e2) into coordinate system
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )
        # Calls Hernquist()
        f_ = self.spherical.function(x_, y_, sigma0, Rs)
        return f_

    def derivatives(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """Returns :math:`\\frac{df}{dx}` and :math:`\\frac{df}{dy}` of the function
        (integral of NFW).

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: gradient of the potential
        """
        # Maps (x, y) with (e1, e2) into coordinate system
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )

        # Convert (e1, e2) to (phi_G, q), which are the orientation angle and
        # axis ratio respectively
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        # Trigonometric components of the rotation
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        # Convert axis ratio q to ellipticity e used for gradient stretching
        e = param_util.q2e(q)

        # Compute the gradient in the transformed (spherical) frame
        f_x_prim, f_y_prim = self.spherical.derivatives(x_, y_, sigma0, Rs)
        # Stretch the gradient components to approximate elliptical effects
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        # Rotate the stretched gradient back to the original (x, y) coordinates
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y

    def hessian(self, x, y, sigma0, Rs, e1, e2, center_x=0, center_y=0):
        """Returns Hessian matrix of function.

        .. math::
            \\frac{d^2f}{dx^2}, \\frac{d^2}{dxdy}, \\frac{d^2}{dydx},
            \\frac{d^f}{dy^2}

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: tuple of second derivatives
        """
        # Evaluate the first derivatives (deflection angles) at (x, y)
        alpha_ra, alpha_dec = self.derivatives(
            x, y, sigma0, Rs, e1, e2, center_x, center_y
        )
        # Small step used for numerical differentiation
        diff = self._diff
        # Evaluate first derivatives at (x + dx, y)
        alpha_ra_dx, alpha_dec_dx = self.derivatives(
            x + diff, y, sigma0, Rs, e1, e2, center_x, center_y
        )
        # Evaluate first derivatives at (x, y + dy)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(
            x, y + diff, sigma0, Rs, e1, e2, center_x, center_y
        )

        # Approximate second derivatives using finite differences
        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_xy, f_yx, f_yy

    def density(self, r, rho0, Rs, e1=0, e2=0):
        """Computes the 3D density.

        :param r: 3D radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: density at radius `r`
        """
        return self.spherical.density(r, rho0, Rs)

    def density_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """Returns the density as a function of 3D radius in lensing parameters.

        This function converts the lensing definition `sigma0` into the 3D
        density.

        :param r: 3D radius
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: enclosed mass in 3D
        """
        return self.spherical.density_lens(r, sigma0, Rs)

    def density_2d(self, x, y, rho0, Rs, e1=0, e2=0, center_x=0, center_y=0):
        """Projected density along the line of sight at coordinate (x, y).

        :param x: x-coordinate
        :param y: y-coordinate
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: x-center of the profile
        :param center_y: y-center of the profile
        :return: projected density
        """
        return self.spherical.density_2d(x, y, rho0, Rs, center_x, center_y)

    def mass_2d_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """Mass enclosed projected 2D sphere of radius `r`. Same as `mass_2d` but with
        input normalization in units of projected density.

        :param r: projected radius
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: mass enclosed 2D projected radius
        """
        return self.spherical.mass_2d_lens(r, sigma0, Rs)

    def mass_2d(self, r, rho0, Rs, e1=0, e2=0):
        """Mass enclosed projected 2D sphere of radius `r`.

        :param r: projected radius
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: mass enclosed 2D projected radius
        """
        return self.spherical.mass_2d(r, rho0, Rs)

    def mass_3d(self, r, rho0, Rs, e1=0, e2=0):
        """Mass enclosed a 3D sphere or radius `r`.

        :param r: 3D radius within the mass is integrated (same distance units as
            density definition)
        :param rho0: density normalization
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: enclosed mass
        """
        return self.spherical.mass_3d(r, rho0, Rs)

    def mass_3d_lens(self, r, sigma0, Rs, e1=0, e2=0):
        """Mass enclosed a 3D sphere or radius `r` in lensing parameterization.

        :param r: 3D radius within the mass is integrated (same distance units as
            density definition)
        :param sigma0: :math:`\\rho_0 \\times R_s` (units of projected density)
        :param Rs: Hernquist radius
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: enclosed mass
        """
        return self.spherical.mass_3d_lens(r, sigma0, Rs)

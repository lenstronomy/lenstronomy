__author__ = "ajshajib"


import numpy as np

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.point_mass import PointMass

__all__ = ["PointMassLogScaled"]


class PointMassLogScaled(LensProfileBase):
    """Point-mass lens profile parameterized by ``log10_theta_E``."""

    param_names = ["log10_theta_E", "center_x", "center_y"]
    lower_limit_default = {"log10_theta_E": -6, "center_x": -100, "center_y": -100}
    upper_limit_default = {"log10_theta_E": 2, "center_x": 100, "center_y": 100}

    def __init__(self):
        self.point_mass = PointMass()
        super(PointMassLogScaled, self).__init__()

    @staticmethod
    def _theta_e(log10_theta_E):
        """Converts log10_theta_E to theta_E.

        :param log10_theta_E: log10 of the Einstein radius (in arcsec)
        :return: Einstein radius (in angles)
        """
        return 10.0**log10_theta_E

    def function(self, x, y, log10_theta_E, center_x=0, center_y=0):
        """Compute the lensing potential at (x, y) for the given log10_theta_E.

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param log10_theta_E: log10 of the Einstein radius (in arcsec)
        :return: lensing potential
        """
        theta_E = self._theta_e(log10_theta_E)
        return self.point_mass.function(
            x, y, theta_E, center_x=center_x, center_y=center_y
        )

    def derivatives(self, x, y, log10_theta_E, center_x=0, center_y=0):
        """Compute the deflection angles at (x, y) for the given log10_theta_E.

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param log10_theta_E: log10 of the Einstein radius (in arcsec)
        :return: deflection angles
        """

        theta_E = self._theta_e(log10_theta_E)
        return self.point_mass.derivatives(
            x, y, theta_E, center_x=center_x, center_y=center_y
        )

    def hessian(self, x, y, log10_theta_E, center_x=0, center_y=0):
        """Compute the Hessian matrix at (x, y) for the given log10_theta_E.

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param log10_theta_E: log10 of the Einstein radius (in arcsec)
        :return: Hessian matrix components
        """
        theta_E = self._theta_e(log10_theta_E)
        return self.point_mass.hessian(
            x, y, theta_E, center_x=center_x, center_y=center_y
        )

    def mass_3d_lens(self, r, log10_theta_E):
        """Compute the 3D mass enclosed within radius r for the given log10_theta_E.

        :param r: radius (in angles)
        :param log10_theta_E: log10 of the Einstein radius (in arcsec)
        :return: 3D mass enclosed within radius r
        """
        theta_E = self._theta_e(log10_theta_E)
        return self.point_mass.mass_3d_lens(r, theta_E)

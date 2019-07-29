import numpy as np
from lenstronomy.LensModel.Profiles.spp import SPP


class CurvedArc(object):
    """
    lens model that describes a section of a highly magnified deflector region.
    The parameterization is chosen to describe local observables efficient.

    Observables are:
    - curvature radius (basically bending relative to the center of the profile)
    - radial stretch (plus sign) thickness of arc with parity (more generalized than the power-law slope)
    - tangential stretch (plus sign). Infinity means at critical curve
    - direction of curvature
    - position of arc

    Requirements:
    - Should work with other perturbative models without breaking its meaning (say when adding additional shear terms)
    - Must best reflect the observables in lensing
    - minimal covariances between the parameters, intuitive parameterization.

    """

    def __init__(self):
        self._spp = SPP()

    def _input2spp_parameterization(self, tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y):
        """

        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param r_curvature: curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return: parameters in terms of a spherical power-law profile resulting in the same observables
        """
        center_x_spp = center_x - r_curvature * np.cos(direction)
        center_y_spp = center_y - r_curvature * np.sin(direction)

        theta_E, gamma = self._stretch2profile(tangential_stretch, radial_stretch, r_curvature)
        return theta_E, gamma, center_x_spp, center_y_spp

    @staticmethod
    def _stretch2profile(tangential_stretch, radial_stretch, r_curvature):
        """

        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param r_curvature: radius of SPP where to have the specific tangential and radial stretch values
        :return: theta_E, gamma of SPP profile
        """
        gamma = (1./radial_stretch - 1) / (1 - 1./tangential_stretch) + 2
        theta_E = abs(1 - 1./tangential_stretch)**(1./(gamma - 1)) * r_curvature
        return theta_E, gamma

    def function(self, x, y, tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch:
        :param radial_stretch:
        :param r_curvature:
        :param direction:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self._input2spp_parameterization(tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y)
        return self._spp.function(x, y, theta_E, gamma, center_x_spp, center_y_spp) - self._spp.function(center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp)

    def derivatives(self, x, y, tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch:
        :param radial_stretch:
        :param r_curvature:
        :param direction:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self._input2spp_parameterization(tangential_stretch,
                                                                                      radial_stretch, r_curvature,
                                                                                      direction, center_x, center_y)
        f_x, f_y = self._spp.derivatives(x, y, theta_E, gamma, center_x_spp, center_y_spp)
        f_x0, f_y0 = self._spp.derivatives(center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp)
        return f_x - f_x0, f_y - f_y0

    def hessian(self, x, y, tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch:
        :param radial_stretch:
        :param r_curvature:
        :param direction:
        :param center_x:
        :param center_y:
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self._input2spp_parameterization(tangential_stretch,
                                                                                      radial_stretch, r_curvature,
                                                                                      direction, center_x, center_y)
        return self._spp.hessian(x, y, theta_E, gamma, center_x_spp, center_y_spp)

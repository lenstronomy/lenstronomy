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
    param_names = ['tangential_stretch', 'radial_stretch', 'r_curvature', 'direction', 'center_x', 'center_y']
    lower_limit_default = {'tangential_stretch': -100, 'radial_stretch': -5, 'r_curvature': 0.001, 'direction': -np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'tangential_stretch': 100, 'radial_stretch': 5, 'r_curvature': 100, 'direction': np.pi, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self._spp = SPP()

    @staticmethod
    def stretch2spp(tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y):
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

        gamma = (1./radial_stretch - 1) / (1 - 1./tangential_stretch) + 2
        theta_E = abs(1 - 1./tangential_stretch)**(1./(gamma - 1)) * r_curvature
        return theta_E, gamma, center_x_spp, center_y_spp

    @staticmethod
    def spp2stretch(theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y):
        """
        turn Singular power-law lens model into stretch parameterization at position (center_x, center_y)
        This is the inverse function of stretch2spp()

        :param theta_E:
        :param gamma:
        :param center_x_spp:
        :param center_y_spp:
        :param center_x:
        :param center_y:
        :return:
        """
        r_curvature = np.sqrt((center_x_spp - center_x)**2 + (center_y_spp - center_y)**2)
        direction = np.arctan2(center_y - center_y_spp, center_x - center_x_spp)
        tangential_stretch = 1 / (1 - (theta_E/r_curvature) ** (gamma - 1))
        radial_stretch = 1 / (1 + (gamma - 2) * (theta_E/r_curvature) ** (gamma - 1))
        return tangential_stretch, radial_stretch, r_curvature, direction

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
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch, radial_stretch, r_curvature, direction, center_x, center_y)
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
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch,
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
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch,
                                                                      radial_stretch, r_curvature,
                                                                      direction, center_x, center_y)
        return self._spp.hessian(x, y, theta_E, gamma, center_x_spp, center_y_spp)

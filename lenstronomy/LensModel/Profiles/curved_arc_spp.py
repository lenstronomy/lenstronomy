import numpy as np
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['CurvedArcSPP']


class CurvedArcSPP(LensProfileBase):
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
    param_names = ['tangential_stretch', 'radial_stretch', 'curvature', 'direction', 'center_x', 'center_y']
    lower_limit_default = {'tangential_stretch': -100, 'radial_stretch': -5, 'curvature': 0.000001, 'direction': -np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'tangential_stretch': 100, 'radial_stretch': 5, 'curvature': 100, 'direction': np.pi, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self._spp = SPP()
        super(CurvedArcSPP, self).__init__()

    @staticmethod
    def stretch2spp(tangential_stretch, radial_stretch, curvature, direction, center_x, center_y):
        """

        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return: parameters in terms of a spherical power-law profile resulting in the same observables
        """
        center_x_spp, center_y_spp = center_deflector(curvature, direction, center_x, center_y)
        r_curvature = 1. / curvature
        gamma = (1./radial_stretch - 1) / (1 - 1./tangential_stretch) + 2
        theta_E = abs(1 - 1./tangential_stretch)**(1./(gamma - 1)) * r_curvature
        return theta_E, gamma, center_x_spp, center_y_spp

    @staticmethod
    def spp2stretch(theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y):
        """
        turn Singular power-law lens model into stretch parameterization at position (center_x, center_y)
        This is the inverse function of stretch2spp()

        :param theta_E: Einstein radius of SPP model
        :param gamma: power-law slope
        :param center_x_spp: center of SPP model
        :param center_y_spp: center of SPP model
        :param center_x: center of curved model definition
        :param center_y: center of curved model definition
        :return: tangential_stretch, radial_stretch, curvature, direction
        """
        r_curvature = np.sqrt((center_x_spp - center_x)**2 + (center_y_spp - center_y)**2)
        direction = np.arctan2(center_y - center_y_spp, center_x - center_x_spp)
        tangential_stretch = 1 / (1 - (theta_E/r_curvature) ** (gamma - 1))
        radial_stretch = 1 / (1 + (gamma - 2) * (theta_E/r_curvature) ** (gamma - 1))
        curvature = 1./r_curvature
        return tangential_stretch, radial_stretch, curvature, direction

    def function(self, x, y, tangential_stretch, radial_stretch, curvature, direction, center_x, center_y):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch, radial_stretch, curvature, direction, center_x, center_y)
        f_ = self._spp.function(x, y, theta_E, gamma, center_x_spp, center_y_spp)
        alpha_x, alpha_y = self._spp.derivatives(center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp)
        f_0 = alpha_x * (x - center_x) + alpha_y * (y - center_y)
        return f_ - f_0

    def derivatives(self, x, y, tangential_stretch, radial_stretch, curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch,
                                                                      radial_stretch, curvature,
                                                                      direction, center_x, center_y)
        f_x, f_y = self._spp.derivatives(x, y, theta_E, gamma, center_x_spp, center_y_spp)
        f_x0, f_y0 = self._spp.derivatives(center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp)
        return f_x - f_x0, f_y - f_y0

    def hessian(self, x, y, tangential_stretch, radial_stretch, curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        theta_E, gamma, center_x_spp, center_y_spp = self.stretch2spp(tangential_stretch,
                                                                      radial_stretch, curvature,
                                                                      direction, center_x, center_y)
        return self._spp.hessian(x, y, theta_E, gamma, center_x_spp, center_y_spp)


def center_deflector(curvature, direction, center_x, center_y):
    """

    :param curvature: 1/curvature radius
    :param direction: float, angle in radian
    :param center_x: center of source in image plane
    :param center_y: center of source in image plane
    :return: center_spp_x, center_spp_y
    """
    center_x_spp = center_x - np.cos(direction) / curvature
    center_y_spp = center_y - np.sin(direction) / curvature
    return center_x_spp, center_y_spp

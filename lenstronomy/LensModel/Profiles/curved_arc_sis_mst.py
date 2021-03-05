import numpy as np
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['CurvedArcSISMST']


class CurvedArcSISMST(LensProfileBase):
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
        self._sis = SIS()
        self._mst = Convergence()
        super(CurvedArcSISMST, self).__init__()

    @staticmethod
    def stretch2sis_mst(tangential_stretch, radial_stretch, curvature, direction, center_x, center_y):
        """

        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return: parameters in terms of a spherical SIS + MST resulting in the same observables
        """
        center_x_sis, center_y_sis = center_deflector(curvature, direction, center_x, center_y)
        r_curvature = 1. / curvature
        lambda_mst = 1./radial_stretch
        kappa_ext = 1 - lambda_mst
        theta_E = r_curvature * (1. - radial_stretch / tangential_stretch)
        return theta_E, kappa_ext, center_x_sis, center_y_sis

    @staticmethod
    def sis_mst2stretch(theta_E, kappa_ext, center_x_sis, center_y_sis, center_x, center_y):
        """
        turn Singular power-law lens model into stretch parameterization at position (center_x, center_y)
        This is the inverse function of stretch2spp()

        :param theta_E: Einstein radius of SIS profile
        :param kappa_ext: external convergence (MST factor 1 - kappa_ext)
        :param center_x_sis: center of SPP model
        :param center_y_sis: center of SPP model
        :param center_x: center of curved model definition
        :param center_y: center of curved model definition
        :return: tangential_stretch, radial_stretch, curvature, direction
        :return:
        """
        r_curvature = np.sqrt((center_x_sis - center_x) ** 2 + (center_y_sis - center_y) ** 2)
        direction = np.arctan2(center_y - center_y_sis, center_x - center_x_sis)
        radial_stretch = 1./ (1 - kappa_ext)
        tangential_stretch = 1 / (1 - (theta_E/r_curvature)) * radial_stretch
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
        lambda_mst = 1. / radial_stretch
        theta_E, kappa_ext, center_x_sis, center_y_sis = self.stretch2sis_mst(tangential_stretch, radial_stretch, curvature, direction, center_x, center_y)
        f_sis = self._sis.function(x, y, theta_E, center_x_sis, center_y_sis)  # - self._sis.function(center_x, center_y, theta_E, center_x_sis, center_y_sis)
        alpha_x, alpha_y = self._sis.derivatives(center_x, center_y, theta_E, center_x_sis, center_y_sis)
        f_sis_0 = alpha_x * (x - center_x) + alpha_y * (y - center_y)
        f_mst = self._mst.function(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        return lambda_mst * (f_sis - f_sis_0) + f_mst

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
        lambda_mst = 1. / radial_stretch
        theta_E, kappa_ext, center_x_sis, center_y_sis = self.stretch2sis_mst(tangential_stretch,
                                                                          radial_stretch, curvature,
                                                                          direction, center_x, center_y)
        f_x_sis, f_y_sis = self._sis.derivatives(x, y, theta_E, center_x_sis, center_y_sis)
        f_x0, f_y0 = self._sis.derivatives(center_x, center_y, theta_E, center_x_sis, center_y_sis)
        f_x_mst, f_y_mst = self._mst.derivatives(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        f_x = lambda_mst * (f_x_sis - f_x0) + f_x_mst
        f_y = lambda_mst * (f_y_sis - f_y0) + f_y_mst
        return f_x, f_y

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
        lambda_mst = 1. / radial_stretch
        theta_E, kappa_ext, center_x_sis, center_y_sis = self.stretch2sis_mst(tangential_stretch,
                                                                          radial_stretch, curvature,
                                                                          direction, center_x, center_y)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = self._sis.hessian(x, y, theta_E, center_x_sis, center_y_sis)
        f_xx_mst, f_xy_mst, f_yx_mst, f_yy_mst = self._mst.hessian(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        return lambda_mst * f_xx_sis + f_xx_mst, lambda_mst * f_xy_sis + f_xy_mst, lambda_mst * f_yx_sis + f_yx_mst, lambda_mst * f_yy_sis + f_yy_mst


def center_deflector(curvature, direction, center_x, center_y):
    """

    :param curvature: 1/curvature radius
    :param direction: float, angle in radian
    :param center_x: center of source in image plane
    :param center_y: center of source in image plane
    :return: center_sis_x, center_sis_y
    """
    center_x_sis = center_x - np.cos(direction) / curvature
    center_y_sis = center_y - np.sin(direction) / curvature
    return center_x_sis, center_y_sis


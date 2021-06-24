import numpy as np
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import param_util

__all__ = ['CurvedArcTanDiff']


class CurvedArcTanDiff(LensProfileBase):
    """
    Curved arc model with an additional non-zero tangential stretch differential in tangential direction component

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
    param_names = ['tangential_stretch', 'radial_stretch', 'curvature', 'dtan_dtan', 'direction', 'center_x', 'center_y']
    lower_limit_default = {'tangential_stretch': -100, 'radial_stretch': -5, 'curvature': 0.000001, 'dtan_dtan': -10, 'direction': -np.pi, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'tangential_stretch': 100, 'radial_stretch': 5, 'curvature': 100, 'dtan_dtab': 10, 'direction': np.pi, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self._sie = SIE(NIE=True)
        self._mst = Convergence()
        super(CurvedArcTanDiff, self).__init__()

    @staticmethod
    def stretch2sie_mst(tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y):
        """

        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
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
        # analytic relation (see Birrer 2021)
        dlambda_tan_dr = tangential_stretch / r_curvature * (1 - tangential_stretch / radial_stretch)

        # translate tangential eigenvalue gradient in lens ellipticity
        dtan_dtan_ = dtan_dtan * tangential_stretch
        epsilon = np.abs(dtan_dtan_ / dlambda_tan_dr)
        # bound epsilon by (-1, 1)
        epsilon = np.minimum(epsilon, 0.99999)
        q = np.sqrt((1 - epsilon) / (1 + epsilon))

        if dtan_dtan_ < 0:
            phi = direction - np.pi / 4
        else:
            phi = direction + np.pi / 4
        e1_sie, e2_sie = param_util.phi_q2_ellipticity(phi, q)

        # ellipticity adopted Einstein radius to match local tangential and radial stretch
        factor = np.sqrt(1 + q ** 2) / np.sqrt(2 * q)
        theta_E_sie = theta_E * factor
        return theta_E_sie, e1_sie, e2_sie, kappa_ext, center_x_sis, center_y_sis

    def function(self, x, y, tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        lambda_mst = 1. / radial_stretch
        theta_E_sie, e1_sie, e2_sie, kappa_ext, center_x_sis, center_y_sis = self.stretch2sie_mst(tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y)
        f_sis = self._sie.function(x, y, theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)  # - self._sis.function(center_x, center_y, theta_E, center_x_sis, center_y_sis)
        alpha_x, alpha_y = self._sie.derivatives(center_x, center_y, theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
        f_sis_0 = alpha_x * (x - center_x) + alpha_y * (y - center_y)
        f_mst = self._mst.function(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        return lambda_mst * (f_sis - f_sis_0) + f_mst

    def derivatives(self, x, y, tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        lambda_mst = 1. / radial_stretch
        theta_E_sie, e1_sie, e2_sie, kappa_ext, center_x_sis, center_y_sis = self.stretch2sie_mst(tangential_stretch,
                                                                              radial_stretch, curvature, dtan_dtan,
                                                                              direction, center_x, center_y)
        f_x_sis, f_y_sis = self._sie.derivatives(x, y, theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
        f_x0, f_y0 = self._sie.derivatives(center_x, center_y, theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
        f_x_mst, f_y_mst = self._mst.derivatives(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        f_x = lambda_mst * (f_x_sis - f_x0) + f_x_mst
        f_y = lambda_mst * (f_y_sis - f_y0) + f_y_mst
        return f_x, f_y

    def hessian(self, x, y, tangential_stretch, radial_stretch, curvature, dtan_dtan, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param dtan_dtan: d(tangential_stretch) / d(tangential direction) / tangential stretch
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        lambda_mst = 1. / radial_stretch
        theta_E_sie, e1_sie, e2_sie, kappa_ext, center_x_sis, center_y_sis = self.stretch2sie_mst(tangential_stretch,
                                                                              radial_stretch, curvature, dtan_dtan,
                                                                              direction, center_x, center_y)
        f_xx_sis, f_xy_sis, f_yx_sis, f_yy_sis = self._sie.hessian(x, y, theta_E_sie, e1_sie, e2_sie, center_x_sis, center_y_sis)
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


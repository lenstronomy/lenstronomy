import numpy as np
from lenstronomy.LensModel.Profiles.shear import ShearReduced
from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['CurvedArcSPT']


class CurvedArcSPT(LensProfileBase):
    """
    Curved arc model based on SIS+MST with an additional non-linear shear distortions applied on the source coordinates
    around the center.
    This profile is effectively a Source Position Transform of a curved arc and a shear distortion.

    """
    param_names = ['tangential_stretch', 'radial_stretch', 'curvature', 'direction', 'gamma1', 'gamma2', 'center_x',
                   'center_y']
    lower_limit_default = {'tangential_stretch': -100, 'radial_stretch': -5, 'curvature': 0.000001, 'direction': -np.pi,
                           'gamma1': -0.5, 'gamma2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'tangential_stretch': 100, 'radial_stretch': 5, 'curvature': 100, 'direction': np.pi,
                           'gamma1': 0.5, 'gamma2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self._curve = CurvedArcSISMST()
        self._distort = ShearReduced()
        super(CurvedArcSPT, self).__init__()

    def function(self, x, y, tangential_stretch, radial_stretch, curvature, direction, gamma1, gamma2, center_x,
                 center_y):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param gamma1: non-linear reduced shear distortion in the source plane
        :param gamma2: non-linear reduced shear distortion in the source plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        raise NotImplemented('lensing potential for regularly curved arc is not implemented')

    def derivatives(self, x, y, tangential_stretch, radial_stretch, curvature, direction, gamma1, gamma2, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param gamma1: non-linear reduced shear distortion in the source plane
        :param gamma2: non-linear reduced shear distortion in the source plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        # computed regular curved arc deflection
        f_x_c, f_y_c = self._curve.derivatives(x, y, tangential_stretch, radial_stretch, curvature, direction,
                                               center_x, center_y)
        # map to source plane coordinate system
        beta_x, beta_y = x - f_x_c, y - f_y_c
        # distort source plane coordinate system around (center_x, center_y)
        f_x_b, f_y_b = self._distort.derivatives(beta_x, beta_y, gamma1, gamma2, ra_0=center_x, dec_0=center_y)
        beta_x_, beta_y_ = beta_x - f_x_b, beta_y - f_y_b
        # compute total deflection between initial coordinate and final source coordinate to match lens equation beta = theta - alpha
        f_x, f_y = x - beta_x_, y - beta_y_
        return f_x, f_y

    def hessian(self, x, y, tangential_stretch, radial_stretch, curvature, direction, gamma1, gamma2, center_x,
                center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param radial_stretch: float, stretch of intrinsic source in radial direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param gamma1: non-linear reduced shear distortion in the source plane
        :param gamma2: non-linear reduced shear distortion in the source plane
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, tangential_stretch, radial_stretch, curvature, direction, gamma1,
                                               gamma2, center_x, center_y)
        diff = 0.0000001
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, tangential_stretch, radial_stretch, curvature,
                                                     direction, gamma1, gamma2, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, tangential_stretch, radial_stretch, curvature,
                                                     direction, gamma1, gamma2, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff
        return f_xx, f_xy, f_yx, f_yy

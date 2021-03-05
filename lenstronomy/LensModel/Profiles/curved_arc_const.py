import numpy as np
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.Util import util

__all__ = ['CurvedArcConstMST']


class CurvedArcConstMST(LensProfileBase):
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
        self._mst = Convergence()
        self._curve = CurvedArcConst()
        super(CurvedArcConstMST, self).__init__()

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
        raise NotImplemented('lensing potential for regularly curved arc is not implemented')

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
        kappa_ext = 1 - lambda_mst
        curve_stretch = tangential_stretch / radial_stretch

        f_x_curve, f_y_curve = self._curve.derivatives(x, y, curve_stretch, curvature, direction, center_x, center_y)
        f_x_mst, f_y_mst = self._mst.derivatives(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        f_x = lambda_mst * f_x_curve + f_x_mst
        f_y = lambda_mst * f_y_curve + f_y_mst
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
        kappa_ext = 1 - lambda_mst
        curve_stretch = tangential_stretch / radial_stretch
        f_xx_c, f_xy_c, f_yx_c, f_yy_c = self._curve.hessian(x, y, curve_stretch, curvature, direction, center_x, center_y)
        f_xx_mst, f_xy_mst, f_yx_mst, f_yy_mst = self._mst.hessian(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        f_xx = lambda_mst * f_xx_c + f_xx_mst
        f_xy = lambda_mst * f_xy_c + f_xy_mst
        f_yx = lambda_mst * f_yx_c + f_yx_mst
        f_yy = lambda_mst * f_yy_c + f_yy_mst
        return f_xx, f_xy, f_yx, f_yy


class CurvedArcConst(LensProfileBase):
    """
    curved arc lensing with orientation of curvature perpendicular to the x-axis with unity radial stretch

    """
    param_names = ['tangential_stretch', 'curvature', 'direction', 'center_x', 'center_y']
    lower_limit_default = {'tangential_stretch': -100, 'curvature': 0.000001, 'direction': -np.pi,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'tangential_stretch': 100, 'curvature': 100, 'direction': np.pi,
                           'center_x': 100, 'center_y': 100}

    def function(self, x, y, tangential_stretch, curvature, direction, center_x, center_y):
        """
        ATTENTION: there may not be a global lensing potential!

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        raise NotImplemented('lensing potential for regularly curved arc is not implemented')

    def derivatives(self, x, y, tangential_stretch, curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """

        r = 1 / curvature
        # deflection angle to allow for tangential stretch
        # (ratio of source position around zero point relative to radius is tangential stretch)
        alpha = r * (1/tangential_stretch + 1)

        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, direction)
        # evaluate
        f__x = alpha * np.cos(y__ * curvature) - alpha
        f__y = alpha * np.sin(y__ * curvature)
        # rotate back
        f_x, f_y = util.rotate(f__x, f__y, -direction)
        return f_x, f_y

    def hessian(self, x, y, tangential_stretch, curvature, direction, center_x, center_y):
        """

        :param x:
        :param y:
        :param tangential_stretch: float, stretch of intrinsic source in tangential direction
        :param curvature: 1/curvature radius
        :param direction: float, angle in radian
        :param center_x: center of source in image plane
        :param center_y: center of source in image plane
        :return:
        """
        alpha_ra, alpha_dec = self.derivatives(x, y, tangential_stretch, curvature, direction, center_x, center_y)
        diff = 0.0000001
        alpha_ra_dx, alpha_dec_dx = self.derivatives(x + diff, y, tangential_stretch, curvature, direction, center_x, center_y)
        alpha_ra_dy, alpha_dec_dy = self.derivatives(x, y + diff, tangential_stretch, curvature, direction, center_x, center_y)

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff

        """
        #TODO make rotational invariances of double derivates with curl
        r = 1 / curvature
        # deflection angle to allow for tangential stretch
        # (ratio of source position around zero point relative to radius is tangential stretch)
        alpha = r * (1 / tangential_stretch + 1)

        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, direction)
        f__xx = 0
        f__xy = -alpha * curvature * np.sin(y__ * curvature)
        f__yx = 0
        f__yy = alpha * curvature * np.cos(y__ * curvature)
        # transform back
        phi_G = direction
        kappa = 1. / 2 * (f__xx + f__yy)
        gamma1__ = 1. / 2 * (f__xx - f__yy)
        gamma2__ = f__xy
        gamma1 = np.cos(2 * phi_G) * gamma1__ - np.sin(2 * phi_G) * gamma2__
        gamma2 = +np.sin(2 * phi_G) * gamma1__ + np.cos(2 * phi_G) * gamma2__
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1

        #f_xx = np.cos(2*direction) * f__xx - np.sin(2*direction) * f__yy
        #f_yy = -np.sin(2*direction) * f__xx + np.cos(2*direction) * f__yy

        f_xy = np.cos(2 * direction) * f__xy - np.sin(2 * direction) * f__yx
        f_yx = -np.sin(2 * direction) * f__xy + np.cos(2 * direction) * f__yx
        return f_xx, f_xy, f_yx, f_yy
        """
        return f_xx, f_xy, f_yx, f_yy

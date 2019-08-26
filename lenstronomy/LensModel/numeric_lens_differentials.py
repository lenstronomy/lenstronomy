from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

from lenstronomy.LensModel.lens_model import LensModel


class NumericLens(LensModel):
    """
    this class computes numerical differentials of lens model quantities
    """
    diff = 0.0000001

    def kappa(self, x, y, kwargs, diff=diff):
        """
        computes the convergence
        :return: kappa
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, diff=diff)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def gamma(self, x, y, kwargs, diff=diff):
        """
        computes the shear
        :return: gamma1, gamma2
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, diff=diff)
        gamma1 = 1./2 * (f_xx - f_yy)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, x, y, kwargs, diff=diff):
        """
        computes the magnification
        :return: potential
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs, diff=diff)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_yx
        return 1/det_A

    def hessian(self, x, y, kwargs, diff=diff):
        """
        computes the differentials f_xx, f_yy, f_xy from f_x and f_y
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra, alpha_dec = self.alpha(x, y, kwargs)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff, y, kwargs)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff, kwargs)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy

    def flexion(self, x, y, kwargs, diff=0.000001):
        """
        third derivatives (flexion)

        :param x: x-position (preferentially arcsec)
        :type x: numpy array
        :param y: y-position (preferentially arcsec)
        :type y: numpy array
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param diff: numerical differential length of Hessian
        :return: f_xxx, f_xxy, f_xyy, f_yyy
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian(x, y, kwargs)

        f_xx_dx, f_xy_dx, f_yx_dx, f_yy_dx = self.hessian(x + diff, y, kwargs)
        f_xx_dy, f_xy_dy, f_yx_dy, f_yy_dy = self.hessian(x, y + diff, kwargs)

        f_xxx = (f_xx_dx - f_xx) / diff
        f_xxy = (f_xx_dy - f_xx) / diff
        f_xyy = (f_xy_dy - f_xy) / diff
        f_yyy = (f_yy_dy - f_yy) / diff
        return f_xxx, f_xxy, f_xyy, f_yyy
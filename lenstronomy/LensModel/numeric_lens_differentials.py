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

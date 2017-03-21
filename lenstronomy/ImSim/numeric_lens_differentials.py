from __future__ import print_function, division, absolute_import, unicode_literals
__author__ = 'sibirrer'

from lenstronomy.ImSim.lens_model import LensModel


class NumericLens(LensModel):
    """
    this class computes numerical differentials of lens model quantities
    """

    def kappa(self, y, x, kwargs_else=None, **kwargs):
        """
        computes the convergence
        :return: kappa
        """
        f_xx, f_xy, f_yx, f_yy = self.differentials(y, x, kwargs_else, **kwargs)
        kappa = 1./2 * (f_xx + f_yy)
        return kappa

    def gamma(self, y, x, kwargs_else=None, **kwargs):
        """
        computes the shear
        :return: gamma1, gamma2
        """
        f_xx, f_xy, f_yx, f_yy = self.differentials(y, x, kwargs_else, **kwargs)
        gamma1 = 1./2 * (f_yy - f_xx)
        gamma2 = f_xy
        return gamma1, gamma2

    def magnification(self, y, x, kwargs_else=None, **kwargs):
        """
        computes the magnification
        :return: potential
        """
        f_xx, f_xy, f_yx, f_yy = self.differentials(y, x, kwargs_else, **kwargs)
        det_A = (1 + f_xx) * (1 + f_yy) - f_xy*f_yx
        return 1/det_A

    def differentials(self, y, x, kwargs_else=None, diff=0.00001, **kwargs):
        """
        computes the differentials f_xx, f_yy, f_xy from f_x and f_y
        :return: f_xx, f_xy, f_yx, f_yy
        """
        alpha_ra, alpha_dec = self.alpha(x, y, kwargs_else, **kwargs)

        alpha_ra_dx, alpha_dec_dx = self.alpha(x + diff, y, kwargs_else, **kwargs)
        alpha_ra_dy, alpha_dec_dy = self.alpha(x, y + diff, kwargs_else, **kwargs)

        dalpha_rara = (alpha_ra_dx - alpha_ra)/diff
        dalpha_radec = (alpha_ra_dy - alpha_ra)/diff
        dalpha_decra = (alpha_dec_dx - alpha_dec)/diff
        dalpha_decdec = (alpha_dec_dy - alpha_dec)/diff

        f_xx = dalpha_rara
        f_yy = dalpha_decdec
        f_xy = dalpha_radec
        f_yx = dalpha_decra
        return f_xx, f_xy, f_yx, f_yy
from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ["PJaffe_Ellipse"]


class PJaffe_Ellipse(LensProfileBase):
    """Class to compute the DUAL PSEUDO ISOTHERMAL ELLIPTICAL MASS DISTRIBUTION based on
    Eliasdottir (2007) https://arxiv.org/pdf/0710.5636.pdf Appendix A with the
    ellipticity implemented in the potential.

    Module name: 'PJAFFE_ELLIPSE';

    An alternative name is dPIED.

    The 3D density distribution is

    .. math::
        \\rho(r) = \\frac{\\rho_0}{(1+r^2/Ra^2)(1+r^2/Rs^2)}

    with :math:`Rs > Ra`.

    The projected density is

    .. math::
        \\Sigma(R) = \\Sigma_0 \\frac{Ra Rs}{Rs-Ra}\\left(\\frac{1}{\\sqrt{Ra^2+R^2}} - \\frac{1}{\\sqrt{Rs^2+R^2}} \\right)

    with

    .. math::
        \\Sigma_0 = \\pi \\rho_0 \\frac{Ra Rs}{Rs + Ra}

    In the lensing parameterization,

    .. math::
        \\sigma_0 = \\frac{\\Sigma_0}{\\Sigma_{\\rm crit}}
    """

    param_names = ["sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "sigma0": 0,
        "Ra": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "sigma0": 10,
        "Ra": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        self.spherical = PJaffe()
        self._diff = 0.000001
        super(PJaffe_Ellipse, self).__init__()

    def function(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns double integral of NFW profile."""
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )
        f_ = self.spherical.function(x_, y_, sigma0, Ra, Rs)
        return f_

    def derivatives(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns df/dx and df/dy of the function (integral of NFW)"""
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_, y_ = param_util.transform_e1e2_square_average(
            x, y, e1, e2, center_x, center_y
        )
        e = param_util.q2e(q)
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        f_x_prim, f_y_prim = self.spherical.derivatives(
            x_, y_, sigma0, Ra, Rs, center_x=0, center_y=0
        )
        f_x_prim *= np.sqrt(1 - e)
        f_y_prim *= np.sqrt(1 + e)
        f_x = cos_phi * f_x_prim - sin_phi * f_y_prim
        f_y = sin_phi * f_x_prim + cos_phi * f_y_prim
        return f_x, f_y

    def hessian(self, x, y, sigma0, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
        d^f/dy^2."""
        alpha_ra, alpha_dec = self.derivatives(
            x, y, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )
        diff = self._diff
        alpha_ra_dx, alpha_dec_dx = self.derivatives(
            x + diff, y, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )
        alpha_ra_dy, alpha_dec_dy = self.derivatives(
            x, y + diff, sigma0, Ra, Rs, e1, e2, center_x, center_y
        )

        f_xx = (alpha_ra_dx - alpha_ra) / diff
        f_xy = (alpha_ra_dy - alpha_ra) / diff
        f_yx = (alpha_dec_dx - alpha_dec) / diff
        f_yy = (alpha_dec_dy - alpha_dec) / diff

        return f_xx, f_xy, f_yx, f_yy

    def mass_3d_lens(self, r, sigma0, Ra, Rs, e1=0, e2=0):
        """:param r:

        :param sigma0:
        :param Ra:
        :param Rs:
        :param e1:
        :param e2:
        :return:
        """
        return self.spherical.mass_3d_lens(r, sigma0, Ra, Rs)

__author__ = 'sibirrer'

import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['Sersic']


class Sersic(SersicUtil, LensProfileBase):
    """
    this class contains functions to evaluate a Sersic mass profile: https://arxiv.org/pdf/astro-ph/0311559.pdf

    .. math::
        \\kappa(R) = \\kappa_{\\rm eff} \\exp \\left[ -b_n (R/R_{\\rm Sersic})^{\\frac{1}{n}}\\right]

    with :math:`b_{n}\\approx 1.999n-0.327`

    Examples for converting physical mass units into convergence units used in the definition of this profile
    ---------------------------------------------------------------------------------------------------------

    We first define an AstroPy cosmology instance and a LensCosmo class instance with a lens and source redshift.

    >>> from lenstronomy.Cosmo.lens_cosmo import LensCosmo
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    >>> lens_cosmo = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)

    We define the half-light radius R_sersic (arc seconds on the sky) and Sersic index n_sersic

    >>> R_sersic = 2
    >>> n_sersic = 4

    Here we compute k_eff, the convergence at the half-light radius R_sersic for a stellar mass in Msun

    >>> k_eff = lens_cosmo.sersic_m_star2k_eff(m_star=10**11.5, R_sersic=R_sersic, n_sersic=n_sersic)

    And here we perform the inverse calculation given k_eff to return the physical stellar mass.

    >>> m_star = lens_cosmo.sersic_k_eff2m_star(k_eff=k_eff, R_sersic=R_sersic, n_sersic=n_sersic)

    The lens model calculation uses angular units as arguments! So to execute a deflection angle calculation one uses

    >>> from lenstronomy.LensModel.Profiles.sersic import Sersic
    >>> sersic = Sersic()
    >>> alpha_x, alpha_y = sersic.derivatives(x=1, y=1, k_eff=k_eff, R_sersic=R_sersic, center_x=0, center_y=0)

    """
    param_names = ['k_eff', 'R_sersic', 'n_sersic', 'center_x', 'center_y']
    lower_limit_default = {'k_eff': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'k_eff': 10, 'R_sersic': 100, 'n_sersic': 8, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, n_sersic, R_sersic, k_eff, center_x=0, center_y=0):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param n_sersic: Sersic index
        :param R_sersic: half light radius
        :param k_eff: convergence at half light radius
        :param center_x: x-center
        :param center_y: y-center
        :return:
        """

        n = n_sersic
        x_red = self._x_reduced(x, y, n, R_sersic, center_x, center_y)
        b = self.b_n(n)
        #hyper2f2_b = util.hyper2F2_array(2*n, 2*n, 1+2*n, 1+2*n, -b)
        hyper2f2_bx = util.hyper2F2_array(2*n, 2*n, 1+2*n, 1+2*n, -b*x_red)
        f_eff = np.exp(b) * R_sersic ** 2 / 2. * k_eff# * hyper2f2_b
        f_ = f_eff * x_red**(2*n) * hyper2f2_bx# / hyper2f2_b
        return f_

    def derivatives(self, x, y, n_sersic, R_sersic, k_eff, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        alpha = -self.alpha_abs(x, y, n_sersic, R_sersic, k_eff, center_x, center_y)
        f_x = alpha * x_ / r
        f_y = alpha * y_ / r
        return f_x, f_y

    def hessian(self, x, y, n_sersic, R_sersic, k_eff, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx, d^f/dy^2
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        d_alpha_dr = self.d_alpha_dr(x, y, n_sersic, R_sersic, k_eff, center_x, center_y)
        alpha = -self.alpha_abs(x, y, n_sersic, R_sersic, k_eff, center_x, center_y)

        f_xx = -(d_alpha_dr/r + alpha/r**2) * x_**2/r + alpha/r
        f_yy = -(d_alpha_dr/r + alpha/r**2) * y_**2/r + alpha/r
        f_xy = -(d_alpha_dr/r + alpha/r**2) * x_*y_/r

        return f_xx, f_xy, f_xy, f_yy

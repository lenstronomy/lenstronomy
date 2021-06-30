import scipy.special as special
import numpy as np
import scipy
from lenstronomy.Util import param_util

__all__ = ['SersicUtil']


class SersicUtil(object):

    _s = 0.00001

    def __init__(self, smoothing=_s, sersic_major_axis=False):
        """

        :param smoothing: smoothing scale of the innermost part of the profile (for numerical reasons)
        :param sersic_major_axis: boolean; if True, defines the half-light radius of the Sersic light profile along
         the semi-major axis (which is the Galfit convention)
         if False, uses the product average of semi-major and semi-minor axis as the convention
         (default definition for all light profiles in lenstronomy other than the Sersic profile)
        """
        self._smoothing = smoothing
        self._sersic_major_axis = sersic_major_axis

    def k_bn(self, n, Re):
        """
        returns normalisation of the sersic profile such that Re is the half light radius given n_sersic slope
        """
        bn = self.b_n(n)
        k = bn*Re**(-1./n)
        return k, bn

    def k_Re(self, n, k):
        """

        """
        bn = self.b_n(n)
        Re = (bn/k)**n
        return Re

    @staticmethod
    def b_n(n):
        """
        b(n) computation. This is the approximation of the exact solution to the relation, 2*incomplete_gamma_function(2n; b_n) = Gamma_function(2*n).
        :param n: the sersic index
        :return:
        """
        bn = 1.9992*n - 0.3271
        bn = np.maximum(bn, 0.00001)  # make sure bn is strictly positive as a save guard for very low n_sersic
        return bn

    def get_distance_from_center(self, x, y, e1, e2, center_x, center_y):
        """
        Get the distance from the center of Sersic, accounting for orientation and axis ratio
        :param x:
        :param y:
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center x of sersic
        :param center_y: center y of sersic
        """

        if self._sersic_major_axis:
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            x_shift = x - center_x
            y_shift = y - center_y
            cos_phi = np.cos(phi_G)
            sin_phi = np.sin(phi_G)
            xt1 = cos_phi*x_shift+sin_phi*y_shift
            xt2 = -sin_phi*x_shift+cos_phi*y_shift
            xt2difq2 = xt2/(q*q)
            r = np.sqrt(xt1*xt1+xt2*xt2difq2)
        else:
            x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
            r = np.sqrt(x_**2 + y_**2)
        return r

    def _x_reduced(self, x, y, n_sersic, r_eff, center_x, center_y):
        """
        coordinate transform to normalized radius
        :param x:
        :param y:
        :param center_x:
        :param center_y:
        :return:
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(self._s, r)
        else:
            r[r < self._s] = self._s
        x_reduced = (r/r_eff)**(1./n_sersic)
        return x_reduced

    def _alpha_eff(self, r_eff, n_sersic, k_eff):
        """
        deflection angle at r_eff
        :param r_eff:
        :param n_sersic:
        :param k_eff:
        :return:
        """
        b = self.b_n(n_sersic)
        alpha_eff = n_sersic * r_eff * k_eff * b**(-2*n_sersic) * np.exp(b) * special.gamma(2*n_sersic)
        return -alpha_eff

    def alpha_abs(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        n = n_sersic
        x_red = self._x_reduced(x, y, n_sersic, r_eff, center_x, center_y)
        b = self.b_n(n_sersic)
        a_eff = self._alpha_eff(r_eff, n_sersic, k_eff)
        alpha = 2. * a_eff * x_red ** (-n) * (special.gammainc(2 * n, b * x_red))
        return alpha

    def d_alpha_dr(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param n_sersic:
        :param r_eff:
        :param k_eff:
        :param center_x:
        :param center_y:
        :return:
        """
        _dr = 0.00001
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        alpha = self.alpha_abs(r, 0, n_sersic, r_eff, k_eff)
        alpha_dr = self.alpha_abs(r+_dr, 0, n_sersic, r_eff, k_eff)
        d_alpha_dr = (alpha_dr - alpha)/_dr
        return d_alpha_dr

    def density(self, x, y, n_sersic, r_eff, k_eff, center_x=0, center_y=0):
        """
        de-projection of the Sersic profile based on
        Prugniel & Simien (1997)
        :return:
        """
        raise ValueError("not implemented! Use a Multi-Gaussian-component decomposition.")

    def _total_flux(self, r_eff, I_eff, n_sersic):
        """
        computes total flux of a round Sersic profile

        :param r_eff: projected half light radius
        :param I_eff: surface brightness at r_eff (in same units as r_eff)
        :param n_sersic: Sersic index
        :return: integrated flux to infinity
        """
        bn = self.b_n(n_sersic)
        return I_eff * r_eff**2 * 2 * np.pi * n_sersic * np.exp(bn) / bn**(2*n_sersic) * scipy.special.gamma(2*n_sersic)

    def total_flux(self, amp, R_sersic, n_sersic, e1=0, e2=0, **kwargs):
        """
        computes analytical integral to compute total flux of the Sersic profile

        :param amp: amplitude parameter in Sersic function (surface brightness at R_sersic
        :param R_sersic: half-light radius in semi-major axis
        :param n_sersic: Sersic index
        :param e1: eccentricity
        :param e2: eccentricity
        :return: Analytic integral of the total flux of the Sersic profile
        """
        # compute product average half-light radius
        if self._sersic_major_axis:
            phi_G, q = param_util.ellipticity2phi_q(e1, e2)
            r_eff = R_sersic * np.sqrt(q)  # translate semi-major axis R_eff into product averaged definition for circularization
        else:
            r_eff = R_sersic
        return self._total_flux(r_eff=r_eff, I_eff=amp, n_sersic=n_sersic)

    def _R_stable(self, R):
        """
        Floor R_ at self._smoothing for numerical stability
        :param R: radius
        :return: smoothed and stabilized radius
        """
        return np.maximum(self._smoothing, R)

    def _r_sersic(self, R, R_sersic, n_sersic, max_R_frac=100.0, alpha=1.0, R_break=0.0):
        """

        :param R: radius (array or float)
        :param R_sersic: Sersic radius (half-light radius)
        :param n_sersic: Sersic index (float)
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: kernel of the Sersic surface brightness at R
        """

        R_ = self._R_stable(R)
        R_sersic_ = self._R_stable(R_sersic)
        bn = self.b_n(n_sersic)
        R_frac = R_ / R_sersic_
        #R_frac = R_frac.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            if R_frac > max_R_frac:
                result = 0
            else:
                exponent = -bn * (R_frac ** (1. / n_sersic) - 1.)
                result = np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= max_R_frac]
            exponent = -bn * (R_frac_real ** (1. / n_sersic) - 1.)
            result = np.zeros_like(R_)
            result[R_frac <= max_R_frac] = np.exp(exponent)
        return np.nan_to_num(result)

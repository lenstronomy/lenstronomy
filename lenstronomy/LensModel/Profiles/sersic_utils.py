import scipy.special as special
import numpy as np
import scipy


class SersicUtil(object):

    _s = 0.000001

    def __init__(self, smoothing=_s):
        self._smoothing = smoothing

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
        b(n) computation
        :param n:
        :return:
        """
        bn = 1.9992 * n - 0.3271
        return bn

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
        computes total flux of a Sersic profile

        :param r_eff: projected half light radius
        :param I_eff: surface brightness at r_eff (in same units as r_eff)
        :param n_sersic: Sersic index
        :return: integrated flux to infinity
        """
        bn = self.b_n(n_sersic)
        return I_eff * r_eff**2 * 2 * np.pi * n_sersic * np.exp(bn) / bn**(2*n_sersic) * scipy.special.gamma(2*n_sersic)

    def total_flux(self, amp, R_sersic, n_sersic, Re=None, gamma=None, e1=None, e2=None, center_x=None, center_y=None,
                   alpha=None):
        """

        :param amp:
        :param R_sersic:
        :param Re:
        :param n_sersic:
        :param gamma:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :param alpha:
        :return:
        """
        return self._total_flux(r_eff=R_sersic, I_eff=amp, n_sersic=n_sersic)

    def _R_stable(self, R):
        """

        :param R: radius
        :return: smoothed and stabilized radius
        """

        if isinstance(R, int) or isinstance(R, float):
            R = max(self._smoothing, R)
        else:
            R[R < self._smoothing] = self._smoothing
        return R

    def _r_sersic(self, R, R_sersic, n_sersic):
        """

        :param R: radius (array or float)
        :param R_sersic: Sersic radius (half-light radius)
        :param n_sersic: Sersic index (float)
        :return: Sersic surface brightness at R
        """

        R_ = self._R_stable(R)
        k, bn = self.k_bn(n_sersic, R_sersic)
        R_frac = R_ / R_sersic
        #R_frac = R_frac.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            if R_frac > 100:
                result = 0
            else:
                exponent = -bn * (R_frac ** (1. / n_sersic) - 1.)
                result = np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= 100]
            exponent = -bn * (R_frac_real ** (1. / n_sersic) - 1.)
            result = np.zeros_like(R_)
            result[R_frac <= 100] = np.exp(exponent)
        return np.nan_to_num(result)

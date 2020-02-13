__author__ = 'sibirrer'

#  this file contains a class to make a Sersic profile

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
import lenstronomy.Util.param_util as param_util


class Sersic(SersicUtil):
    """
    this class contains functions to evaluate an spherical Sersic function
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


class SersicElliptic(SersicUtil):
    """
    this class contains functions to evaluate an elliptical Sersic function
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_sersic': 0, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5,'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'R_sersic': 100, 'n_sersic': 8, 'e1': 0.5, 'e2': 0.5,'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0, max_R_frac=100.0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param n_sersic: Sersic index
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Sersic profile value at (x, y)
        """

        R_sersic = np.maximum(0, R_sersic)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R_ = np.sqrt(xt1*xt1+xt2*xt2difq2)
        result = self._r_sersic(R_, R_sersic, n_sersic, max_R_frac)
        return amp * result


class CoreSersic(SersicUtil):
    """
    this class contains the Core-Sersic function introduced by e.g Trujillo et al. 2004
    """
    param_names = ['amp', 'R_sersic', 'Re', 'n_sersic', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'Re': 0, 'n_sersic': 0.5, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'amp': 100, 'Re': 100, 'n_sersic': 8, 'gamma': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def function(self, x, y, amp, R_sersic, Re, n_sersic, gamma, e1, e2, center_x=0, center_y=0, alpha=3.0, max_R_frac=100.0):
        """
        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_sersic: semi-major axis half light radius
        :param Re: "break" core radius
        :param n_sersic: Sersic index
        :param gamma: inner power-law exponent
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param alpha: sharpness of the transition between the cusp and the outer Sersic profile (float)
        :param max_R_frac: maximum window outside of which the mass is zeroed, in units of R_sersic (float)
        :return: Cored Sersic profile value at (x, y)
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        Rb = R_sersic
        x_shift = x - center_x
        y_shift = y - center_y

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R_ = np.sqrt(xt1*xt1+xt2*xt2difq2)
        R_ = self._R_stable(R_)
        R = np.maximum(self._smoothing, R_) # floor R_ at self._smoothing for numerical stability
        k, bn = self.k_bn(n_sersic, Re)
        result = amp * (1 + (Rb / R) ** alpha) ** (gamma / alpha) * np.exp(-bn * (((R ** alpha + Rb ** alpha) / Re ** alpha) ** (1. / (alpha * n_sersic)) - 1.))
        return np.nan_to_num(result)

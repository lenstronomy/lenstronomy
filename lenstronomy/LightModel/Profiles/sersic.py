__author__ = 'sibirrer'

#this file contains a class to make a sersic profile

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
import lenstronomy.Util.param_util as param_util


class Sersic(SersicUtil):
    """
    this class contains functions to evaluate an spherical Sersic function
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'center_x', 'center_y']

    def function(self, x, y, amp, R_sersic, n_sersic, center_x=0, center_y=0):
        """
        returns Sersic profile
        """
        #if n_sersic < 0.2:
        #    n_sersic = 0.2
        #if R_sersic < 10.**(-6):
        #    R_sersic = 10.**(-6)
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            R = max(self._smoothing, R)
        else:
            R[R < self._smoothing] = self._smoothing
        _, bn = self.k_bn(n_sersic, R_sersic)
        R_frac = R/R_sersic
        #R_frac = R_frac.astype(np.float32)
        if isinstance(R, int) or isinstance(R, float):
            if R_frac > 100:
                result = 0
            else:
                exponent = -bn*(R_frac**(1./n_sersic)-1.)
                result = amp * np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= 100]
            exponent = -bn*(R_frac_real**(1./n_sersic)-1.)
            result = np.zeros_like(R)
            result[R_frac <= 100] = amp * np.exp(exponent)
        return result


class Sersic_elliptic(SersicUtil):
    """
    this class contains functions to evaluate an elliptical Sersic function
    """
    param_names = ['amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y']

    def function(self, x, y, amp, R_sersic, n_sersic, e1, e2, center_x=0, center_y=0):
        """
        returns Sersic profile
        """
        #if n_sersic < 0.2:
        #    n_sersic = 0.2
        #if R_sersic < 10.**(-6):
        #    R_sersic = 10.**(-6)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        x_shift = x - center_x
        y_shift = y - center_y

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        xt1 = cos_phi*x_shift+sin_phi*y_shift
        xt2 = -sin_phi*x_shift+cos_phi*y_shift
        xt2difq2 = xt2/(q*q)
        R_ = np.sqrt(xt1*xt1+xt2*xt2difq2)
        if isinstance(R_, int) or isinstance(R_, float):
            R_ = max(self._smoothing, R_)
        else:
            R_[R_ < self._smoothing] = self._smoothing
        k, bn = self.k_bn(n_sersic, R_sersic)
        R_frac = R_/R_sersic
        R_frac = R_frac.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            if R_frac > 100:
                result = 0
            else:
                exponent = -bn*(R_frac**(1./n_sersic)-1.)
                result = amp * np.exp(exponent)
        else:
            R_frac_real = R_frac[R_frac <= 100]
            exponent = -bn*(R_frac_real**(1./n_sersic)-1.)
            result = np.zeros_like(R_)
            result[R_frac <= 100] = amp * np.exp(exponent)
        return result


class CoreSersic(SersicUtil):
    """
    this class contains the Core-Sersic function introduced by e.g Trujillo et al. 2004
    """
    param_names = ['amp', 'R_sersic', 'Re', 'n_sersic', 'gamma', 'e1', 'e2', 'center_x', 'center_y']

    def function(self, x, y, amp, R_sersic, Re, n_sersic, gamma, e1, e2, center_x=0, center_y=0, alpha=3.):
        """
        returns Core-Sersic function
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
        #R_ = R_.astype(np.float32)
        if isinstance(R_, int) or isinstance(R_, float):
            R_ = max(self._smoothing, R_)
        else:
            R_[R_ < self._smoothing] = self._smoothing
        if isinstance(R_, int) or isinstance(R_, float):
            R = max(self._smoothing, R_)
        else:
            R=np.empty_like(R_)
            _R = R_[R_ > self._smoothing]  #in the SIS regime
            R[R_ <= self._smoothing] = self._smoothing
            R[R_ > self._smoothing] = _R

        k, bn = self.k_bn(n_sersic, Re)
        return amp * (1 + (Rb / R) ** alpha) ** (gamma / alpha) * np.exp(-bn * (((R ** alpha + Rb ** alpha) / Re ** alpha) ** (1. / (alpha * n_sersic)) - 1.))

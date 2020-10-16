__author__ = 'sibirrer'

#  this file contains a class to make a Sersic profile

import numpy as np
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
import lenstronomy.Util.param_util as param_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
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
        R = self.get_distance_from_center(x, y, phi_G=0.0, q=1.0, center_x=center_x, center_y=center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


@export
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
        R = self.get_distance_from_center(x, y, phi_G, q, center_x, center_y)
        result = self._r_sersic(R, R_sersic, n_sersic, max_R_frac)
        return amp * result


@export
class CoreSersic(SersicUtil):
    """
    this class contains the Core-Sersic function introduced by e.g Trujillo et al. 2004
    """
    param_names = ['amp', 'R_sersic', 'Re', 'n_sersic', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'Re': 0, 'n_sersic': 0.5, 'gamma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'amp': 100, 'Re': 100, 'n_sersic': 8, 'gamma': 10, 'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def function(self, x, y, amp, R_sersic, Re, n_sersic, gamma, e1, e2, center_x=0, center_y=0, alpha=3.0,
                 max_R_frac=100.0):
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
        #TODO max_R_frac not implemented
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        R_ = self.get_distance_from_center(x, y, phi_G, q, center_x, center_y)
        R = self._R_stable(R_)
        bn = self.b_n(n_sersic)
        result = amp * (1 + (Re / R)**alpha)**(gamma/alpha)*np.exp(-bn*(((R ** alpha + Re ** alpha)/R_sersic**alpha)**(1./(alpha*n_sersic)) - 1.))
        return np.nan_to_num(result)

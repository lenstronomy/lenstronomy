from lenstronomy.LightModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.chameleon import Chameleon as ChameleonLens
import lenstronomy.Util.param_util as param_util
import numpy as np


class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['amp', 'w_c', 'w_t', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'w_c': 0, 'w_t': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'w_c': 100, 'w_t': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.nie = NIE()
        self._chameleonLens = ChameleonLens()

    def function(self, x, y, amp, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param amp: amplitude of first power-law flux
        :param flux_ratio: ratio of amplitudes of first to second power-law profile
        :param gamma1: power-law slope
        :param gamma2: power-law slope
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: center
        :param center_y: center
        :return: flux of chameleon profile
        """
        amp_new, w_c, w_t = self._chameleonLens._theta_convert(amp, w_c, w_t)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        flux1 = self.nie.function(x, y, 1, e1, e2, s_scale_1, center_x, center_y)
        flux2 = self.nie.function(x, y, 1, e1, e2, s_scale_2, center_x, center_y)
        flux = amp_new / (1. + q) * (flux1 - flux2)
        return flux


class DoubleChameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['amp', 'ratio', 'w_c1', 'w_t1', 'e11', 'e21', 'w_c2', 'w_t2', 'e12', 'e22', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'ratio': 0, 'w_c1': 0, 'w_t1': 0, 'e11': -0.8, 'e21': -0.8,
                           'w_c2': 0, 'w_t2': 0, 'e12': -0.8, 'e22': -0.8,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'ratio': 100, 'w_c1': 100, 'w_t1': 100, 'e11': 0.8, 'e21': 0.8,
                           'w_c2': 100, 'w_t2': 100, 'e12': 0.8, 'e22': 0.8,
                           'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.chameleon = Chameleon()

    def function(self, x, y, amp, ratio, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, center_x=0, center_y=0):
        """

        :param amp:
        :param ratio:
        :param w_c1:
        :param w_t1:
        :param e11:
        :param e21:
        :param w_c2:
        :param w_t2:
        :param e12:
        :param e22:
        :param center_x:
        :param center_y:
        :return:
        """
        f_1 = self.chameleon.function(x, y, amp / (1. + 1./ratio), w_c1, w_t1, e11, e21, center_x, center_y)
        f_2 = self.chameleon.function(x, y, amp / (1. + ratio), w_c2, w_t2, e12, e22, center_x, center_y)
        return f_1 + f_2


class TripleChameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['amp', 'ratio12', 'ratio13', 'w_c1', 'w_t1', 'e11', 'e21',
                   'w_c2', 'w_t2', 'e12', 'e22', 'w_c3', 'w_t3', 'e13',
                   'e23', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'ratio12': 0, 'ratio13': 0.,
                           'w_c1': 0, 'w_t1': 0, 'e11': -0.8, 'e21': -0.8,
                           'w_c2': 0, 'w_t2': 0, 'e12': -0.8, 'e22': -0.8,
                           'w_c3': 0, 'w_t3': 0, 'e13': -0.8, 'e23': -0.8,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'ratio12': 100, 'ratio13': 100,
                           'w_c1': 100, 'w_t1': 100, 'e11': 0.8, 'e21': 0.8,
                           'w_c2': 100, 'w_t2': 100, 'e12': 0.8, 'e22': 0.8,
                           'w_c3': 100, 'w_t3': 100, 'e13': 0.8, 'e23': 0.8,
                           'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.chameleon = Chameleon()

    def function(self, x, y, amp, ratio12, ratio13, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, w_c3, w_t3, e13, e23,
                 center_x=0, center_y=0):
        """

        :param amp:
        :param ratio12: ratio of first to second amplitude
        :param ratio13: ratio of first to third amplidute
        :param w_c1:
        :param w_t1:
        :param e11:
        :param e21:
        :param w_c2:
        :param w_t2:
        :param e12:
        :param e22:
        :param center_x:
        :param center_y:
        :return:
        """
        amp1 = amp / (1. + 1./ratio12 + 1./ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        f_1 = self.chameleon.function(x, y, amp1, w_c1, w_t1, e11, e21, center_x, center_y)
        f_2 = self.chameleon.function(x, y, amp2, w_c2, w_t2, e12, e22, center_x, center_y)
        f_3 = self.chameleon.function(x, y, amp3, w_c2, w_t2, e12, e22, center_x, center_y)
        return f_1 + f_2 + f_3

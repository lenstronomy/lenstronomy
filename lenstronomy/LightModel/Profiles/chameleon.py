from lenstronomy.LightModel.Profiles.nie import NIE
import lenstronomy.Util.param_util as param_util
import numpy as np


class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['amp', 'w_c', 'w_t', 'e1', 'e2', 'center_x', 'center_y']

    def __init__(self):
        self.nie = NIE()

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
        if not w_t > w_c:
            w_t, w_c = w_c, w_t
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        flux1 = self.nie.function(x, y, 1, e1, e2, s_scale_1, center_x, center_y)
        flux2 = self.nie.function(x, y, 1, e1, e2, s_scale_2, center_x, center_y)
        flux = amp / (1. + q) * (flux1 - flux2)
        return flux

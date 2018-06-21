import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.spp import SPP
import numpy as np
import scipy.special as special


class PowerLaw(object):
    """
    class for power-law elliptical mass distribution

    """
    param_names = ['amp', 'gamma', 'e1', 'e2', 'center_x', 'center_y']

    def __init__(self):
        self.lens = SPP()

    def function(self, x, y, amp, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param gamma: projected power-law slope
        :param e1: ellipticity
        :param e2: ellipticity
        :param center_x: center
        :param center_y: center
        :return: projected flux
        """
        x_, y_ = param_util.transform_e1e2(x, y, e1, e2, center_x, center_y)
        P2 = x_ ** 2 + y_ ** 2
        if isinstance(P2, int) or isinstance(P2, float):
            a = max(0.00000001, P2)
        else:
            a = np.empty_like(P2)
            p2 = P2[P2 > 0]  # in the SIS regime
            a[P2 == 0] = 0.00000001
            a[P2 > 0] = p2

        sigma = amp * a ** ((1. - gamma)/2.)
        return sigma

    def light_3d(self, r, amp, gamma, e1, e2):
        """

        :param r:
        :param amp:
        :param gamma:
        :param e1:
        :param e2:
        :return:
        """
        rho0 = self._amp2rho(amp, gamma)
        rho = rho0 / r ** gamma
        return rho

    def _amp2rho(self, amp, gamma):
        """

        :param amp:
        :param gamma:
        :return:
        """
        factor = np.sqrt(np.pi) * special.gamma(1./2*(-1+gamma))/special.gamma(gamma/2.)
        return amp / factor

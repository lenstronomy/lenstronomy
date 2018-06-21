import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util


class NIE(object):
    """
    non-divergent isothermal ellipse (projected)
    """
    param_names = ['amp', 'e1', 'e2', 's_scale', 'center_x', 'center_y']

    def function(self, x, y, amp, e1, e2, s_scale, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param e1:
        :param e2:
        :param s_scale:
        :param center_x:
        :param center_y:
        :return:
        """
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        # shift
        x_ = x - center_x
        y_ = y - center_y
        # rotate
        x__, y__ = util.rotate(x_, y_, phi_G)
        # evaluate
        f_ = self._nie_simple_function(x__, y__, amp, s_scale, q)
        # rotate back
        return f_

    def _nie_simple_function(self, x, y, amp, s, q):
        """

        :param x:
        :param y:
        :param amp:
        :param s_cale:
        :param q:
        :return:
        """
        return amp / np.sqrt(q**2*(s**2 + x**2) + y**2)
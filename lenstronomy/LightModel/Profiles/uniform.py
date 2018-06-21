import numpy as np


class Uniform(object):
    """
    class for Gaussian light profile
    """
    param_names = ['amp']

    def __init__(self):
        pass

    def function(self, x, y, amp):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        return np.ones_like(x) * amp


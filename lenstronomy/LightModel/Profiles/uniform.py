import numpy as np


class Uniform(object):
    """
    class for Gaussian light profile
    """
    param_names = ['amp']
    lower_limit_default = {'amp': -100}
    upper_limit_default = {'amp': 100}

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

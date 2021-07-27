import numpy as np

__all__ = ['Uniform']


class Uniform(object):
    """
    uniform light profile. This profile can also compensate for an inaccurate background subtraction.
    name for profile: 'UNIFORM'
    """
    param_names = ['amp']
    lower_limit_default = {'amp': -100}
    upper_limit_default = {'amp': 100}

    def __init__(self):
        pass

    def function(self, x, y, amp):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :return: constant flux
        """
        return np.ones_like(x) * amp

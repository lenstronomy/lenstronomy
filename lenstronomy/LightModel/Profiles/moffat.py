__author__ = 'sibirrer'

# this file contains a class to make a moffat profile

__all__ = ['Moffat']


class Moffat(object):
    """
    this class contains functions to evaluate a Moffat surface brightness profile

    .. math::

        I(r) = I_0 * (1 + (r/\alpha)^2)^{-\beta}

    with :math:`I_0 = amp`.

    """
    def __init__(self):
        self.param_names = ['amp', 'alpha', 'beta', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'alpha': 0, 'beta': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 100, 'alpha': 10, 'beta': 10, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, alpha, beta, center_x=0, center_y=0):
        """
        2D Moffat profile

        :param x: x-position (angle)
        :param y: y-position (angle)
        :param amp: normalization
        :param alpha: scale
        :param beta: exponent
        :param center_x: x-center
        :param center_y: y-center
        :return: surface brightness
        """

        x_shift = x - center_x
        y_shift = y - center_y
        return amp * (1. + (x_shift**2+y_shift**2)/alpha**2)**(-beta)

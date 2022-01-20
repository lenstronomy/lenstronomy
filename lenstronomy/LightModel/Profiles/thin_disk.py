__author__ = 'martin-millon'

#  this file contains a class to make a thin disk profile (Shakura & Sunyaev 1973)
import numpy as np

__all__ = ['ThinDisk']

class ThinDisk(object):
    """
    this class contains functions to evaluate an thin disk model

    .. math::
        I(R) = I_0 / \\left[\\exp(\\xi) - 1\\right]

    with :math:`I_0 = amp`
        and
    with :math:`\\xi (R) = \\left[ R / R_0 \\right] ^{3/4}  \\left[ 1 - \\sqrt{R_in/R}\\right] ^{-1/4}`

    """

    param_names = ['amp', 'R_0', 'R_in', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_0': 0.0, 'R_in': 0.0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100000, 'R_0': 100, 'R_in': 10, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_0, R_in, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_0: scale radius
        :param R_in: inner radius of the profile
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :return: Thin disk profile value at (x, y)
        """
        R = np.sqrt((x - center_x) ** 2 + (y - center_y)**2)
        profile = np.zeros_like(R)
        if R_in <= 0:
            profile = amp / (np.exp((R / R_0) ** (0.75)) - 1)
        else:
            profile = np.where(R <= R_in, profile,
                               amp / (np.exp((R / R_0) ** (0.75) * (1 - np.sqrt(R_in / R)) ** (-0.25)) - 1))

        return profile

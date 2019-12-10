__author__ = 'dgilman'

import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

class FreeForm(LensProfileBase):
    """
    this class contains the function and the derivatives of a free form lens model with user-specified deflections and
    potential at image positions
    """
    param_names = ['alpha_x', 'alpha_y']
    lower_limit_default = {'alpha_x': -10, 'alpha_y': -10}
    upper_limit_default = {'alpha_x': 10, 'alpha_y': 10}

    def function(self, x, y, potential, alpha_x, alpha_y, center_x=0, center_y=0):

        """
        This is currently not implemented, and will default to 0

        :param x: x position to evaluate potential
        :param y: y position to evaluate potential
        :param potential: value of the potential at (x, y)
        :param center_x: not used
        :param center_y: not used
        :return: lensing potential at image positions
        """

        return potential

    def derivatives(self, x, y, potential, alpha_x, alpha_y, center_x=0, center_y=0):
        """

        :param x: x position to evaluate deflection angle
        :param y: y position to evaluate deflection angle
        :param alpha_x: deflection at x-coordinates
        :param alpha_y: deflection at y-coordinates
        :param center_x: not used
        :param center_y: not used
        :return: deflections at image positions
        """
        return np.array(alpha_x), np.array(alpha_y)

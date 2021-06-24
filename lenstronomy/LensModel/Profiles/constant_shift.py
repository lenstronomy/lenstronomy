__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ['Shift']


class Shift(LensProfileBase):
    """
    Lens model with a constant shift of the deflection field
    """
    param_names = ['alpha_x', 'alpha_y']
    lower_limit_default = {'alpha_x': -1000, 'alpha_y': -1000}
    upper_limit_default = {'alpha_x': 1000, 'alpha_y': 1000}

    def function(self, x, y, alpha_x, alpha_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param alpha_x: shift in x-direction (angle)
        :param alpha_y: shift in y-direction (angle)
        :return: lensing potential
        """

        return np.zeros_like(x)

    def derivatives(self, x, y, alpha_x, alpha_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param alpha_x: shift in x-direction (angle)
        :param alpha_y: shift in y-direction (angle)
        :return: deflection in x- and y-direction
        """
        f_x = np.ones_like(x) * alpha_x
        f_y = np.ones_like(x) * alpha_y
        return f_x, f_y

    def hessian(self, x, y, alpha_x, alpha_y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param alpha_x: shift in x-direction (angle)
        :param alpha_y: shift in y-direction (angle)
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        f_xx, f_xy, f_yx, f_yy = 0, 0, 0, 0
        return f_xx, f_xy, f_yx, f_yy

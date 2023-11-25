__author__ = 'ajshajib'

from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
import numpy as np

__all__ = ['BlankPlane']


class BlankPlane(LensProfileBase):
    """
    Class for a blank lens plane. This is needed to a create a blank plane
    that has a source without any lensing effect, when distance ratios are
    sampled in multi-lens-plane and multi-source plane case.
    """
    param_names = []
    lower_limit_default = {}
    upper_limit_default = {}

    def function(self, x, y):
        """
        """
        return np.zeros_like(x)

    def derivatives(self, x, y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param alpha_x: shift in x-direction (angle)
        :param alpha_y: shift in y-direction (angle)
        :return: deflection in x- and y-direction
        """
        return np.zeros_like(x), np.zeros_like(x)

    def hessian(self, x, y):
        """

        :param x: coordinate in image plane (angle)
        :param y: coordinate in image plane (angle)
        :param alpha_x: shift in x-direction (angle)
        :param alpha_y: shift in y-direction (angle)
        :return: hessian elements f_xx, f_xy, f_yx, f_yy
        """
        return np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

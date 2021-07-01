__author__ = 'sibirrer'

#this file contains a class to make a moffat profile
import numpy as np
from lenstronomy.Util import param_util

__all__ = ['Ellipsoid']


class Ellipsoid(object):
    """
    class for an universal surface brightness within an ellipsoid
    """
    def __init__(self):
        self.param_names = ['amp', 'radius', 'e1', 'e2', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'radius': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 1000, 'radius': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, radius, e1, e2, center_x, center_y):
        """

        :param x:
        :param y:
        :param amp: surface brightness within the ellipsoid
        :param radius: radius (product average of semi-major and semi-minor axis) of the ellipsoid
        :param e1: eccentricity
        :param e2: eccentricity
        :param center_x: center
        :param center_y: center
        :return: surface brightness
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        r2 = x_**2 + y_**2
        flux = np.zeros_like(x)
        flux[r2 <= radius**2] = 1
        A = np.pi * radius ** 2
        return amp / A * flux


def function(x, y, amp, sigma, center_x, center_y):
    """
    returns torus (ellipse with constant surface brightness) profile
    """
    x_shift = x - center_x
    y_shift = y - center_y
    A = np.pi * sigma**2
    dist = (x_shift / sigma) ** 2 + (y_shift / sigma) ** 2
    torus = np.zeros_like(x)
    torus[dist <= 1] = 1
    return amp/A * torus

__author__ = 'sibirrer'

#this file contains a class to make a moffat profile
import numpy as np


class Ellipsoid(object):
    """
    class for an universal surface brightness within an ellipsoid
    """
    def function(self, x, y, amp, radius, e1, e2, center_x, center_y):
        """

        :param x:
        :param y:
        :param amp:
        :param radius:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :return:
        """
        pass



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

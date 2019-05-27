__author__ = 'sibirrer'

#this file contains a class to make a moffat profile
import numpy as np


def function(x, y, amp, a_x, a_y, center_x, center_y):
    """
    returns torus (ellipse with constant surface brightnes) profile
    """
    x_shift = x - center_x
    y_shift = y - center_y
    A = np.pi * a_x * a_y
    dist = (x_shift/a_x)**2 + (y_shift/a_y)**2
    torus = np.zeros_like(x)
    torus[dist <= 1] = 1
    return amp/A * torus

"""
routines to compute derivatives of spherical functions
"""
import numpy as np


def d_r_dx(x, y):
    """
    derivative of r with respect to x
    :param x:
    :param y:
    :return:
    """
    return x / np.sqrt(x**2 + y**2)


def d_r_dy(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    return y / np.sqrt(x**2 + y**2)


def d_x_diffr_dx(x, y):
    """
    derivative of d(x/r)/dx
    :param x:
    :param y:
    :return:
    """
    return y**2 / (x**2 + y**2)**(3/2.)


def d_y_diffr_dy(x, y):
    """
    derivative of d(y/r)/dy
    :param x:
    :param y:
    :return:
    """
    return x**2 / (x**2 + y**2)**(3/2.)


def d_y_diffr_dx(x, y):
    """
    derivative of d(y/r)/dx
    :param x:
    :param y:
    :return:
    """
    return -x*y / (x**2 + y**2)**(3/2.)


def d_x_diffr_dy(x, y):
    """
    derivative of d(x/r)/dy
    :param x:
    :param y:
    :return:
    """
    return -x*y / (x**2 + y**2)**(3/2.)

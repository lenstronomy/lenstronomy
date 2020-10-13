"""
routines to compute derivatives of spherical functions
"""
import numpy as np

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def d_r_dx(x, y):
    """
    derivative of r with respect to x
    :param x:
    :param y:
    :return:
    """
    return x / np.sqrt(x**2 + y**2)


@export
def d_r_dy(x, y):
    """
    differential dr/dy

    :param x:
    :param y:
    :return:
    """
    return y / np.sqrt(x**2 + y**2)


@export
def d_r_dxx(x, y):
    """
    second derivative dr/dxdx
    :param x:
    :param y:
    :return:
    """
    return y**2 / (x**2 + y**2)**(3./2)


@export
def d_r_dyy(x, y):
    """
    second derivative dr/dxdx
    :param x:
    :param y:
    :return:
    """
    return x**2 / (x**2 + y**2)**(3./2)


@export
def d_r_dxy(x, y):
    """
    second derivative dr/dxdx
    :param x:
    :param y:
    :return:
    """
    return -x * y / (x ** 2 + y ** 2) ** (3 / 2.)


@export
def d_phi_dx(x, y):
    """
    angular derivative in respect to x when phi = arctan2(y, x)

    :param x:
    :param y:
    :return:
    """
    return -y / (x**2 + y**2)


@export
def d_phi_dy(x, y):
    """
    angular derivative in respect to y when phi = arctan2(y, x)

    :param x:
    :param y:
    :return:
    """
    return x / (x**2 + y**2)


@export
def d_phi_dxx(x, y):
    """
    second derivative of the orientation angle

    :param x:
    :param y:
    :return:
    """
    return 2 * x * y / (x**2 + y**2)**2


@export
def d_phi_dyy(x, y):
    """
    second derivative of the orientation angle in dydy

    :param x:
    :param y:
    :return:
    """
    return -2 * x * y / (x ** 2 + y ** 2) ** 2


@export
def d_phi_dxy(x, y):
    """
    second derivative of the orientation angle in dxdy

    :param x:
    :param y:
    :return:
    """
    return (-x**2 + y**2) / (x ** 2 + y ** 2) ** 2


@export
def d_x_diffr_dx(x, y):
    """
    derivative of d(x/r)/dx
    equivalent to second order derivatives dr_dxx

    :param x:
    :param y:
    :return:
    """
    return y**2 / (x**2 + y**2)**(3/2.)


@export
def d_y_diffr_dy(x, y):
    """
    derivative of d(y/r)/dy
    equivalent to second order derivatives dr_dyy

    :param x:
    :param y:
    :return:
    """
    return x**2 / (x**2 + y**2)**(3/2.)


@export
def d_y_diffr_dx(x, y):
    """
    derivative of d(y/r)/dx
    equivalent to second order derivatives dr_dxy

    :param x:
    :param y:
    :return:
    """
    return -x*y / (x**2 + y**2)**(3/2.)


@export
def d_x_diffr_dy(x, y):
    """
    derivative of d(x/r)/dy
    equivalent to second order derivatives dr_dyx

    :param x:
    :param y:
    :return:
    """
    return -x*y / (x**2 + y**2)**(3/2.)

import numpy as np

from lenstronomy.Util.numba_util import jit
from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
def cart2polar(x, y, center_x=0, center_y=0):
    """
    transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the lens center

    :param x: set of x-coordinates
    :type x: array of size (n)
    :param y: set of x-coordinates
    :type y: array of size (n)
    :param center_x: rotation point
    :type center_x: float
    :param center_y: rotation point
    :type center_y: float
    :returns:  array of same size with coords [r,phi]
    """
    coord_shift_x = x - center_x
    coord_shift_y = y - center_y
    r = np.sqrt(coord_shift_x**2+coord_shift_y**2)
    phi = np.arctan2(coord_shift_y, coord_shift_x)
    return r, phi


@export
def polar2cart(r, phi, center):
    """
    transforms polar coords [r,phi] into cartesian coords [x,y] in the frame of the lense center

    :param coord: set of coordinates
    :type coord: array of size (n,2)
    :param center: rotation point
    :type center: array of size (2)
    :returns:  array of same size with coords [x,y]
    :raises: AttributeError, KeyError
    """
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x - center[0], y - center[1]


@export
def shear_polar2cartesian(phi, gamma):
    """

    :param phi: shear angle (radian)
    :param gamma: shear strength
    :return: shear components gamma1, gamma2
    """
    gamma1 = gamma*np.cos(2*phi)
    gamma2 = gamma*np.sin(2*phi)
    return gamma1, gamma2


@export
def shear_cartesian2polar(gamma1, gamma2):
    """
    :param gamma1: cartesian shear component
    :param gamma2: cartesian shear component
    :return: shear angle, shear strength
    """
    phi = np.arctan2(gamma2, gamma1) / 2
    gamma = np.sqrt(gamma1 ** 2 + gamma2 ** 2)
    return phi, gamma


@export
@jit()
def phi_q2_ellipticity(phi, q):
    """
    transforms orientation angle and axis ratio into complex ellipticity moduli e1, e2

    :param phi: angle of orientation (in radian)
    :param q: axis ratio minor axis / major axis
    :return: eccentricities e1 and e2 in complex ellipticity moduli
    """
    e1 = (1. - q) / (1. + q) * np.cos(2 * phi)
    e2 = (1. - q) / (1. + q) * np.sin(2 * phi)
    return e1, e2


@export
@jit()
def ellipticity2phi_q(e1, e2):
    """
    transforms complex ellipticity moduli in orientation angle and axis ratio

    :param e1: eccentricity in x-direction
    :param e2: eccentricity in xy-direction
    :return: angle in radian, axis ratio (minor/major)
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    c = np.minimum(c, 0.9999)
    q = (1-c)/(1+c)
    return phi, q


@export
def transform_e1e2_product_average(x, y, e1, e2, center_x, center_y):
    """
    maps the coordinates x, y with eccentricities e1 e2 into a new elliptical coordinate system
    such that R = sqrt(R_major * R_minor)

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y

    cos_phi = np.cos(phi_G)
    sin_phi = np.sin(phi_G)

    xt1 = cos_phi * x_shift + sin_phi * y_shift
    xt2 = -sin_phi * x_shift + cos_phi * y_shift
    return xt1 * np.sqrt(q), xt2 / np.sqrt(q)


@export
def transform_e1e2_square_average(x, y, e1, e2, center_x, center_y):
    """
    maps the coordinates x, y with eccentricities e1 e2 into a new elliptical coordinate system
    such that R = sqrt(R_major**2 + R_minor**2)

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    phi_G, q = ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y
    cos_phi = np.cos(phi_G)
    sin_phi = np.sin(phi_G)
    e = abs(1 - q)
    x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
    y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
    return x_, y_

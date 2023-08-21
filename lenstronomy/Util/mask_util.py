import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.Util import param_util

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def mask_center_2d(center_x, center_y, r, x_grid, y_grid):
    """

    :param center_x: x-coordinate of center position of circular mask
    :param center_y: y-coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param x_grid: x-coordinate grid
    :param y_grid: y-coordinate grid
    :return: mask array of shape x_grid with =0 inside the radius and =1 outside
    :rtype: array of size of input grid with integers 0 or 1
    """
    x_shift = x_grid - center_x
    y_shift = y_grid - center_y
    R = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    mask = np.empty_like(R, dtype="int")
    mask[R > r] = 1
    mask[R <= r] = 0
    return mask


@export
def mask_azimuthal(x, y, center_x, center_y, r):
    """Azimuthal mask with =1 inside radius and =0 outside.

    :param x: x-coordinates (1d or 2d array numpy array)
    :param y: y-coordinates (1d or 2d array numpy array)
    :param center_x: center of azimuthal mask in x
    :param center_y: center of azimuthal mask in y
    :param r: radius of azimuthal mask
    :return: array with zeros outside r and ones inside azimuthal radius r
    :rtype: array of size of input grid with integers 0 or 1
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    mask = np.empty_like(R, dtype="int")
    mask[R > r] = 0
    mask[R <= r] = 1
    return mask


@export
def mask_ellipse(x, y, center_x, center_y, a, b, angle):
    """

    :param x: x-coordinates of pixels
    :param y: y-coordinates of pixels
    :param center_x: center of mask
    :param center_y: center of mask
    :param a: major axis
    :param b: minor axis
    :param angle: angle of major axis
    :return: mask (list of zeros and ones)
    :rtype: array of size of input grid with integers 0 or 1
    """
    x_shift = x - center_x
    y_shift = y - center_y
    x_rot, y_rot = util.rotate(x_shift, y_shift, angle)
    r_ab = x_rot**2 / a**2 + y_rot**2 / b**2
    mask = np.empty_like(r_ab, dtype="int")
    mask[r_ab > 1] = 0
    mask[r_ab <= 1] = 1
    return mask


@export
def mask_eccentric(x, y, center_x, center_y, e1, e2, r):
    """Elliptical mask with eccentricities as input.

    :param x: x-coordinate array
    :param y: y-coordinate array
    :param center_x: center in x
    :param center_y: center in y
    :param e1: eccentricity e1
    :param e2: eccentricity e2
    :param r: radius of mask
    :return: mask (list of zeros and ones)
    :rtype: array of size of input grid with integers 0 or 1
    """
    x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
    r_ab = (x_**2 + y_**2) / r**2
    mask = np.empty_like(r_ab, dtype="int")
    mask[r_ab > 1] = 0
    mask[r_ab <= 1] = 1
    return mask


@export
def mask_half_moon(x, y, center_x, center_y, r_in, r_out, phi0=0, delta_phi=2 * np.pi):
    """

    :param x:
    :param y:
    :param center_x:
    :param center_y:
    :param r_in:
    :param r_out:
    :param phi0:
    :param delta_phi:
    :return:
    :rtype: array of size of input grid with integers 0 or 1
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    phi = np.arctan2(x_shift, y_shift)
    phi_min = phi0 - delta_phi / 2.0
    phi_max = phi0 + delta_phi / 2.0
    mask = np.zeros_like(x, dtype="int")
    if phi_max > phi_min:
        mask[(R < r_out) & (R > r_in) & (phi > phi_min) & (phi < phi_max)] = 1
    else:
        mask[(R < r_out) & (R > r_in) & (phi > phi_max)] = 1
        mask[(R < r_out) & (R > r_in) & (phi < phi_min)] = 1
    return mask


def mask_shell(x, y, center_x, center_y, r_in, r_out):
    """Ring mask.

    :param x: x-coordinate grid
    :param y: y-coordinate grid
    :param center_x: center of ring mask
    :param center_y: center of ring mask
    :param r_in: inner ring radius
    :param r_out: outer ring radius
    :return: mask
    """
    x_shift = x - center_x
    y_shift = y - center_y
    r = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    mask = np.zeros_like(x, dtype="int")
    mask[(r >= r_in) & (r < r_out)] = 1
    return mask

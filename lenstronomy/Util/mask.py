import numpy as np
import lenstronomy.Util.util as util


def mask_center_2d(center_x, center_y, r, x, y):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.empty_like(R)
    mask[R > r] = 1
    mask[R <= r] = 0
    n = int(np.sqrt(len(x)))
    mask_2d = mask.reshape(n, n)
    return mask_2d


def mask_sphere(x, y, center_x, center_y, r):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    mask = np.empty_like(R)
    mask[R > r] = 0
    mask[R <= r] = 1
    return mask


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
    """
    x_shift = x - center_x
    y_shift = y - center_y
    x_rot, y_rot = util.rotate(x_shift, y_shift, angle)
    r_ab = x_rot**2 / a**2 + y_rot**2 / b**2
    mask = np.empty_like(r_ab)
    mask[r_ab > 1] = 0
    mask[r_ab <= 1] = 1
    return mask

def mask_half_moon(x, y, center_x, center_y, r_in, r_out, phi0=0, delta_phi=2*np.pi):
    """

    :param x:
    :param y:
    :param center_x:
    :param center_y:
    :param r_in:
    :param r_out:
    :param phi:
    :param delta_phi:
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
    phi = np.arctan2(x_shift, y_shift)
    #phi = np.abs(phi)
    phi_min = phi0 - delta_phi/2.
    phi_max = phi0 + delta_phi/2.
    mask = np.zeros_like(x)
    if phi_max > phi_min:
        mask[(R < r_out) & (R > r_in) & (phi > phi_min) & (phi < phi_max)] = 1
    else:
        mask[(R < r_out) & (R > r_in) & (phi > phi_max)] = 1
        mask[(R < r_out) & (R > r_in) & (phi < phi_min)] = 1
    return mask
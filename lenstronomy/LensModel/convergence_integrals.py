import numpy as np
import scipy.signal as scp
from lenstronomy.Util import util
from lenstronomy.Util import image_util
from lenstronomy.Util import kernel_util

"""
class to compute lensing potentials and deflection angles provided a convergence map
"""

from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def potential_from_kappa_grid(kappa, grid_spacing, plummer_softening_length=None):
    """Lensing potential :math:`\\psi ({\\vec {\\theta }})` on the convergence grid
    :math:`\\kappa`.

    .. math::
        \\psi ({\\vec {\\theta }})={\\frac {1}{\\pi }}\\int d^{2}\\theta ^{\\prime }
        \\kappa ({\\vec {\\theta }}^{\\prime })\\ln |{\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime }|

    The computation is performed as a convolution of the Green's function with the convergence map using FFT.
    If the Plummer softening length is provided, the kernel is modified so that each pixel is treated as a Plummer sphere.

    :param kappa: 2d grid of convergence values
    :param grid_spacing: scale of an individual pixel (per axis) of grid
    :param plummer_softening_length: optional, Plummer softening length in the same units of delta_pix.
    :return: lensing potential in a 2d grid at positions x_grid, y_grid
    """
    num_pix = len(kappa) * 2
    if num_pix % 2 == 0:
        num_pix += 1
    kernel = potential_kernel(num_pix, grid_spacing, plummer_softening_length)
    f_ = scp.fftconvolve(kappa, kernel, mode="same") / np.pi * grid_spacing**2
    return f_


@export
def potential_from_kappa_grid_adaptive(
    kappa_high_res,
    grid_spacing,
    low_res_factor,
    high_res_kernel_size,
    plummer_softening_length=None,
):
    """Lensing potential on the convergence grid the computation is performed as a
    convolution of the Green's function with the convergence map using FFT.

    If the Plummer softening length is provided, the kernel is modified so that each
    pixel is treated as a Plummer sphere.

    :param kappa_high_res: 2d grid of convergence values
    :param grid_spacing: scale of an individual pixel (per axis) of grid
    :param low_res_factor: lower resolution factor of larger scale kernel.
    :param high_res_kernel_size: int, size of high resolution kernel in units of
        degraded pixels
    :param plummer_softening_length: optional, Plummer softening length in the same
        units of delta_pix.
    :return: lensing potential in a 2d grid at positions x_grid, y_grid
    """
    kappa_low_res = image_util.re_size(kappa_high_res, factor=low_res_factor)
    num_pix = len(kappa_high_res) * 2
    if num_pix % 2 == 0:
        num_pix += 1
    grid_spacing_low_res = grid_spacing * low_res_factor
    kernel = potential_kernel(num_pix, grid_spacing, plummer_softening_length)
    kernel_low_res, kernel_high_res = kernel_util.split_kernel(
        kernel, high_res_kernel_size, low_res_factor
    )
    kernel_low_res /= low_res_factor**2
    # , normalized = False

    f_high_res = (
        scp.fftconvolve(kappa_high_res, kernel_high_res, mode="same")
        / np.pi
        * grid_spacing**2
    )
    f_high_res = image_util.re_size(f_high_res, low_res_factor)
    f_low_res = (
        scp.fftconvolve(kappa_low_res, kernel_low_res, mode="same")
        / np.pi
        * grid_spacing_low_res**2
    )
    return f_high_res + f_low_res


@export
def deflection_from_kappa_grid(kappa, grid_spacing, plummer_softening_length=None):
    """Deflection angle :math:`\\vec {\\alpha }}` from a convergence grid
    :math:`\\kappa`.

    .. math::
        {\\vec {\\alpha }}({\\vec {\\theta }})={\\frac {1}{\\pi }}
        \\int d^{2}\\theta ^{\\prime }{\\frac {({\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime })
        \\kappa ({\\vec {\\theta }}^{\\prime })}{|{\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime }|^{2}}}

    The computation is performed as a convolution of the Green's function with the convergence map using FFT.
    If the Plummer softening length is provided, the kernel is modified so that each pixel is treated as a Plummer sphere.

    :param kappa: convergence values for each pixel (2-d array)
    :param grid_spacing: scale of an individual pixel (per axis) of grid
    :param plummer_softening_length: optional, Plummer softening length in the same units of delta_pix.
    :return: numerical deflection angles in x- and y- direction over the convergence grid points
    """
    num_pix = len(kappa) * 2
    if num_pix % 2 == 0:
        num_pix += 1
    kernel_x, kernel_y = deflection_kernel(
        num_pix, grid_spacing, plummer_softening_length
    )
    f_x = scp.fftconvolve(kappa, kernel_x, mode="same") / np.pi * grid_spacing**2
    f_y = scp.fftconvolve(kappa, kernel_y, mode="same") / np.pi * grid_spacing**2
    return f_x, f_y


@export
def deflection_from_kappa_grid_adaptive(
    kappa_high_res,
    grid_spacing,
    low_res_factor,
    high_res_kernel_size,
    plummer_softening_length=None,
):
    """Deflection angles on the convergence grid with adaptive FFT the computation is
    performed as a convolution of the Green's function with the convergence map using
    FFT. The grid is returned in the lower resolution grid.

    If the Plummer softening length is provided, the kernel is modified so that each
    pixel is treated as a Plummer sphere.

    :param kappa_high_res: convergence values for each pixel (2-d array)
    :param grid_spacing: pixel size of high resolution grid
    :param low_res_factor: lower resolution factor of larger scale kernel.
    :param high_res_kernel_size: int, size of high resolution kernel in units of
        degraded pixels
    :param plummer_softening_length: optional, Plummer softening length in the same
        units of delta_pix.
    :return: numerical deflection angles in x- and y- direction
    """
    kappa_low_res = image_util.re_size(kappa_high_res, factor=low_res_factor)
    num_pix = len(kappa_high_res) * 2
    if num_pix % 2 == 0:
        num_pix += 1

    kernel_x, kernel_y = deflection_kernel(
        num_pix, grid_spacing, plummer_softening_length
    )
    grid_spacing_low_res = grid_spacing * low_res_factor

    kernel_low_res_x, kernel_high_res_x = kernel_util.split_kernel(
        kernel_x, high_res_kernel_size, low_res_factor
    )
    kernel_low_res_x /= low_res_factor**2
    # , normalized=False
    f_x_high_res = (
        scp.fftconvolve(kappa_high_res, kernel_high_res_x, mode="same")
        / np.pi
        * grid_spacing**2
    )
    f_x_high_res = image_util.re_size(f_x_high_res, low_res_factor)
    f_x_low_res = (
        scp.fftconvolve(kappa_low_res, kernel_low_res_x, mode="same")
        / np.pi
        * grid_spacing_low_res**2
    )
    f_x = f_x_high_res + f_x_low_res

    kernel_low_res_y, kernel_high_res_y = kernel_util.split_kernel(
        kernel_y, high_res_kernel_size, low_res_factor, normalized=False
    )
    f_y_high_res = (
        scp.fftconvolve(kappa_high_res, kernel_high_res_y, mode="same")
        / np.pi
        * grid_spacing**2
    )
    f_y_high_res = image_util.re_size(f_y_high_res, low_res_factor)
    f_y_low_res = (
        scp.fftconvolve(kappa_low_res, kernel_low_res_y, mode="same")
        / np.pi
        * grid_spacing_low_res**2
    )
    f_y = f_y_high_res + f_y_low_res
    return f_x, f_y


@export
def potential_kernel(num_pix, delta_pix, plummer_softening_length=None):
    """Numerical gridded integration kernel for convergence to lensing kernel with given
    pixel size.

    If the Plummer softening length is provided, the kernel is modified so that each
    pixel is treated as a Plummer sphere.

    :param num_pix: integer; number of pixels of kernel per axis
    :param delta_pix: pixel size (per dimension in units of angle)
    :param plummer_softening_length: optional, Plummer softening length in the same
        units of delta_pix.
    :return: kernel for lensing potential
    """
    x_shift, y_shift = util.make_grid(num_pix=num_pix, delta_pix=delta_pix)
    r2 = x_shift**2 + y_shift**2
    if plummer_softening_length is None:
        r2_max = np.max(r2)
        r2[r2 < (delta_pix / 2) ** 2] = (delta_pix / 2) ** 2
        kernel = np.log(r2 / r2_max) / 2.0
    else:
        kernel = np.log(1 + (r2 / plummer_softening_length**2)) / 2

    kernel = util.array2image(kernel)
    return kernel


@export
def deflection_kernel(num_pix, delta_pix, plummer_softening_length=None):
    """Numerical gridded integration kernel for convergence to deflection angle with
    given pixel size.

    If the Plummer softening length is provided, the kernel is modified so that each
    pixel is treated as a Plummer sphere.

    :param num_pix: integer; number of pixels of kernel per axis, should be odd number
        to have a defined center
    :param delta_pix: pixel size (per dimension in units of angle)
    :param plummer_softening_length: optional, Plummer softening length in the same
        units of delta_pix.
    :return: kernel for x-direction and kernel of y-direction deflection angles
    """
    x_shift, y_shift = util.make_grid(num_pix=num_pix, delta_pix=delta_pix)
    r2 = x_shift**2 + y_shift**2
    if plummer_softening_length is None:
        r2[r2 < (delta_pix / 2) ** 2] = (delta_pix / 2) ** 2
        kernel_x = x_shift / r2
        kernel_y = y_shift / r2
    else:
        kernel_x = x_shift / (r2 + plummer_softening_length**2)
        kernel_y = y_shift / (r2 + plummer_softening_length**2)

    return util.array2image(kernel_x), util.array2image(kernel_y)

import numpy as np
import scipy.signal as scp
from lenstronomy.Util import util
"""
class to compute lensing potentials and deflection angles provided a convergence map
"""

#TODO adaptive map and kernel
#TODO real space kernel with adaptive mesh refinement


def potential_from_kappa_grid(kappa, grid_spacing):
    """
    lensing potential on the convergence grid
    the computation is performed as a convolution of the Green's function with the convergence map using FFT

    :param kappa: 2d grid of convergence values
    :param grid_spacing: pixel size of grid
    :return: lensing potential in a 2d grid at positions x_grid, y_grid
    """
    num_pix = len(kappa) * 2
    kernel = potential_kernel(num_pix, grid_spacing)
    f_ = scp.fftconvolve(kappa, kernel, mode='same') / np.pi * grid_spacing ** 2
    return f_


def deflection_from_kappa_grid(kappa, grid_spacing):
    """
    deflection angles on the convergence grid
    the computation is performed as a convolution of the Green's function with the convergence map using FFT

    :param kappa: convergence values for each pixel (2-d array)
    :param grid_spacing: pixel size of grid
    :return: numerical deflection angles in x- and y- direction
    """
    num_pix = len(kappa) * 2
    kernel_x, kernel_y = deflection_kernel(num_pix, grid_spacing)
    f_x = scp.fftconvolve(kappa, kernel_x, mode='same') / np.pi * grid_spacing ** 2
    f_y = scp.fftconvolve(kappa, kernel_y, mode='same') / np.pi * grid_spacing ** 2
    return f_x, f_y


def potential_kernel(num_pix, delta_pix):
    """
    numerical gridded integration kernel for convergence to lensing kernel with given pixel size

    :param num_pix: integer; number of pixels of kernel per axis
    :param delta_pix: pixel size (per dimension in units of angle)
    :return: kernel for lensing potential
    """
    x_shift, y_shift = util.make_grid(numPix=num_pix, deltapix=delta_pix)
    r2 = x_shift ** 2 + y_shift ** 2
    r2_max = np.max(r2)
    r2[r2 < (delta_pix / 2) ** 2] = (delta_pix / 2) ** 2
    lnr = np.log(r2/r2_max) / 2.
    kernel = util.array2image(lnr)
    return kernel


def deflection_kernel(num_pix, delta_pix):
    """
    numerical gridded integration kernel for convergence to deflection angle with given pixel size

    :param num_pix: integer; number of pixels of kernel per axis
    :param delta_pix: pixel size (per dimension in units of angle)
    :return: kernel for x-direction and kernel of y-direction deflection angles
    """
    x_shift, y_shift = util.make_grid(numPix=num_pix, deltapix=delta_pix)
    r2 = x_shift**2 + y_shift**2
    r2[r2 < (delta_pix/2)**2] = (delta_pix/2) ** 2

    kernel_x = util.array2image(x_shift / r2)
    kernel_y = util.array2image(y_shift / r2)
    return kernel_x, kernel_y

"""
routines that manipulate convolution kernels
"""
import numpy as np
import copy
import scipy.ndimage.interpolation as interp
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util


def de_shift_kernel(kernel, shift_x, shift_y, iterations=20):
    """

    :param kernel:
    :param shift_x:
    :param shift_y:
    :return:
    """
    n = len(kernel)
    kernel_new = np.zeros((n+2, n+2)) + (kernel[0, 0] + kernel[0, -1] + kernel[-1, 0] + kernel[-1, -1]) / 4.
    kernel_new[1:-1, 1:-1] = kernel
    int_shift_x = int(round(shift_x))
    frac_x_shift = shift_x - int_shift_x
    int_shift_y = int(round(shift_y))
    frac_y_shift = shift_y - int_shift_y
    kernel_init = copy.deepcopy(kernel_new)
    kernel_init_shifted = copy.deepcopy(interp.shift(kernel_init, [int_shift_y, int_shift_x], order=1))
    kernel_new = interp.shift(kernel_new, [int_shift_y, int_shift_x], order=1)
    norm = np.sum(kernel_new)
    for i in range(iterations):
        kernel_shifted_inv = interp.shift(kernel_new, [-frac_y_shift, -frac_x_shift], order=1)
        delta = kernel_init_shifted - kernel_norm(kernel_shifted_inv) * norm
        kernel_new += delta
        kernel_new = kernel_norm(kernel_new) * norm
    return kernel_new[1:-1, 1:-1]


def kernel_norm(kernel):
    """

    :param kernel:
    :return: normalisation of the psf kernel
    """
    norm = np.sum(np.array(kernel))
    kernel /= norm
    return kernel


def subgrid_kernel(kernel, subgrid_res, odd=False):
    """
    creates a higher resolution kernel with subgrid resolution as an interpolation of the original kernel
    :param kernel: initial kernel
    :param subgrid_res: subgrid resolution required
    :return: kernel with higher resolution (larger)
        """
    numPix = len(kernel)
    x_in = np.linspace(0, 1, numPix)
    numPix_new = numPix * subgrid_res
    if odd is True:
        if numPix_new % 2 == 0:
            numPix_new -= 1
    x_out = np.linspace(0, 1, numPix_new)
    out_values = image_util.re_size_array(x_in, x_in, kernel, x_out, x_out)
    kernel_subgrid = out_values
    kernel_subgrid = kernel_norm(kernel_subgrid)
    return kernel_subgrid


def kernel_pixelsize_change(kernel, deltaPix_in, deltaPix_out):
    """
    change the pixel size of a given kernel
    :param kernel:
    :param deltaPix_in:
    :param deltaPix_out:
    :return:
    """
    numPix = len(kernel)
    numPix_new = int(round(numPix * deltaPix_in/deltaPix_out))
    if numPix_new % 2 == 0:
        numPix_new -= 1
    x_in = np.linspace(-(numPix-1)/2*deltaPix_in, (numPix-1)/2*deltaPix_in, numPix)
    x_out = np.linspace(-(numPix_new-1)/2*deltaPix_out, (numPix_new-1)/2*deltaPix_out, numPix_new)
    kernel_out = image_util.re_size_array(x_in, x_in, kernel, x_out, x_out)
    kernel_out = kernel_norm(kernel_out)
    return kernel_out


def cut_psf(psf_data, psf_size):
    """
    cut the psf properly
    :param psf_data: image of PSF
    :param psf_size: size of psf
    :return: re-sized and re-normalized PSF
    """
    kernel = image_util.cut_edges(psf_data, psf_size)
    kernel = kernel_norm(kernel)
    return kernel


def pixel_kernel(point_source_kernel, subgrid_res=7):
    """
    converts a pixelised kernel of a point source to a kernel representing a uniform extended pixel
    :param point_source_kernel:
    :param subgrid_res:
    :return: convolution kernel for an extended pixel
    """
    kernel_subgrid = subgrid_kernel(point_source_kernel, subgrid_res)
    kernel_size = len(point_source_kernel)
    kernel_pixel = np.zeros((kernel_size*subgrid_res, kernel_size*subgrid_res))
    for i in range(subgrid_res):
        k_x = int((kernel_size-1) / 2 * subgrid_res + i)
        for j in range(subgrid_res):
            k_y = int((kernel_size-1) / 2 * subgrid_res + j)
            kernel_pixel = image_util.add_layer2image(kernel_pixel, k_x, k_y, kernel_subgrid)
    kernel_pixel = util.averaging(kernel_pixel, numGrid=kernel_size*subgrid_res, numPix=kernel_size)
    return kernel_norm(kernel_pixel)


def cutout_source(x_pos, y_pos, image, kernelsize, shift=True):
    """
    cuts out point source (e.g. PSF estimate) out of image and shift it to the center of a pixel
    :param x_pos:
    :param y_pos:
    :param image:
    :param kernelsize:
    :return:
    """
    if kernelsize%2 == 0:
        raise ValueError("even pixel number kernel size not supported!")
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    n = len(image)
    d = (kernelsize - 1)/2
    x_max = np.minimum(x_int + d + 1, n)
    x_min = np.maximum(x_int - d, 0)
    y_max = np.minimum(y_int + d + 1, n)
    y_min = np.maximum(y_int - d, 0)
    image_cut = copy.deepcopy(image[y_min:y_max, x_min:x_max])
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    if shift is True:
        kernel_final = de_shift_kernel(image_cut, shift_x, shift_y)
    else:
        kernel_final = image_cut
    return kernel_final


def fwhm_kernel(kernel):
    """
    computes the full width at half maximum of a (PSF) kernel
    :param kernel: (psf) kernel, 2d numpy array
    :return: fwhm in units of pixels
    """
    n = len(kernel)
    if n % 2 == 0:
        raise ValueError('only works with odd number of pixels in kernel!')
    max_flux = kernel[(n-1)/2, (n-1)/2]
    I_2 = max_flux/2.
    I_r = kernel[(n-1)/2, (n-1)/2:]
    r = np.linspace(0, (n-1)/2, (n+1)/2)
    for i in range(1, len(r)):
        if I_r[i] < I_2:
            fwhm_2 = (I_2 - I_r[i-1])/(I_r[i] - I_r[i-1]) + r[i-1]
            return fwhm_2 * 2
    raise ValueError('The kernel did not drop to half the max value - fwhm not determined!')

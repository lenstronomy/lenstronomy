"""
routines that manipulate convolution kernels
"""
import numpy as np
import copy
import scipy.ndimage.interpolation as interp
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian


def de_shift_kernel(kernel, shift_x, shift_y, iterations=20):
    """

    :param kernel:
    :param shift_x:
    :param shift_y:
    :return:
    """
    nx, ny = np.shape(kernel)
    kernel_new = np.zeros((nx+2, ny+2)) + (kernel[0, 0] + kernel[0, -1] + kernel[-1, 0] + kernel[-1, -1]) / 4.
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
    subgrid_res = int(subgrid_res)
    if subgrid_res == 1:
        return kernel
    nx, ny = np.shape(kernel)
    d_x = 1. / nx
    x_in = np.linspace(d_x/2, 1-d_x/2, nx)
    d_y = 1. / nx
    y_in = np.linspace(d_y/2, 1-d_y/2, ny)
    nx_new = nx * subgrid_res
    ny_new = ny * subgrid_res
    if odd is True:
        if nx_new % 2 == 0:
            nx_new -= 1
        if ny_new % 2 == 0:
            ny_new -= 1

    d_x_new = 1. / nx_new
    d_y_new = 1. / ny_new
    x_out = np.linspace(d_x_new/2, 1-d_x_new/2, nx_new)
    y_out = np.linspace(d_y_new/2, 1-d_y_new, ny_new)
    out_values = image_util.re_size_array(x_in, y_in, kernel, x_out, y_out)
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


def kernel_gaussian(kernel_numPix, deltaPix, fwhm):
    sigma = util.fwhm2sigma(fwhm)
    x_grid, y_grid = util.make_grid(kernel_numPix, deltaPix)
    gaussian = Gaussian()
    kernel = gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma, sigma_y=sigma,
                                         center_x=0, center_y=0)
    kernel /= np.sum(kernel)
    kernel = util.array2image(kernel)
    return kernel


def cutout_source(x_pos, y_pos, image, kernelsize, shift=True):
    """
    cuts out point source (e.g. PSF estimate) out of image and shift it to the center of a pixel
    :param x_pos:
    :param y_pos:
    :param image:
    :param kernelsize:
    :return:
    """
    if kernelsize % 2 == 0:
        raise ValueError("even pixel number kernel size not supported!")
    x_int = int(round(x_pos))
    y_int = int(round(y_pos))
    n = len(image)
    d = (kernelsize - 1)/2
    x_max = int(np.minimum(x_int + d + 1, n))
    x_min = int(np.maximum(x_int - d, 0))
    y_max = int(np.minimum(y_int + d + 1, n))
    y_min = int(np.maximum(y_int - d, 0))
    image_cut = copy.deepcopy(image[y_min:y_max, x_min:x_max])
    shift_x = x_int - x_pos
    shift_y = y_int - y_pos
    if shift is True:
        kernel_shift = de_shift_kernel(image_cut, shift_x, shift_y)
    else:
        kernel_shift = image_cut
    kernel_final = np.zeros((kernelsize, kernelsize))

    k_l2_x = int((kernelsize - 1) / 2)
    k_l2_y = int((kernelsize - 1) / 2)

    xk_min = np.maximum(0, -x_int + k_l2_x)
    yk_min = np.maximum(0, -y_int + k_l2_y)
    xk_max = np.minimum(kernelsize, -x_int + k_l2_x + n)
    yk_max = np.minimum(kernelsize, -y_int + k_l2_y + n)

    kernel_final[yk_min:yk_max, xk_min:xk_max] = kernel_shift
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
    max_flux = kernel[int((n-1)/2), int((n-1)/2)]
    I_2 = max_flux/2.
    I_r = kernel[int((n-1)/2), int((n-1)/2):]
    r = np.linspace(0, (n-1)/2, (n+1)/2)
    for i in range(1, len(r)):
        if I_r[i] < I_2:
            fwhm_2 = (I_2 - I_r[i-1])/(I_r[i] - I_r[i-1]) + r[i-1]
            return fwhm_2 * 2
    raise ValueError('The kernel did not drop to half the max value - fwhm not determined!')


def estimate_amp(data, x_pos, y_pos, psf_kernel):
    """
    estimates the amplitude of a point source located at x_pos, y_pos
    :param data:
    :param x_pos:
    :param y_pos:
    :param deltaPix:
    :return:
    """
    numPix_x, numPix_y = np.shape(data)
    #data_center = int((numPix-1.)/2)
    x_int = int(round(x_pos-0.49999))#+data_center
    y_int = int(round(y_pos-0.49999))#+data_center
    if x_int > 2 and x_int < numPix_x-2 and y_int > 2 and y_int < numPix_y-2:
        mean_image = max(np.sum(data[y_int-2:y_int+3, x_int-2:x_int+3]), 0)
        num = len(psf_kernel)
        center = int((num-0.5)/2)
        mean_kernel = np.sum(psf_kernel[center-2:center+3, center-2:center+3])
        amp_estimated = mean_image/mean_kernel
    else:
        amp_estimated = 0
    return amp_estimated
import numpy as np
import copy

from lenstronomy.Util import numba_util
from lenstronomy.ImSim.Numerics.partial_image import PartialImage

"""
the convolution is inspired by pyautolens: https://github.com/Jammy2211/PyAutoLens
"""


class NumbaConvolution(object):
    """
    class to convolve explicit pixels only
    """
    def __init__(self, kernel, conv_pixels, compute_pixels=None, nopython=True, cache=True, parallel=False):
        """

        :param kernel: convolution kernel in units of the image pixels provided, odd length per axis
        :param conv_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        :param nopython: bool, numba jit setting to use python or compiled.
        :param cache: bool, numba jit setting to use cache
        :param parallel: bool, numba jit setting to use parallel mode
        """
        numba_util.nopython = nopython
        numba_util.cache = cache
        numba_util.parallel = parallel

        self._kernel = kernel
        self._conv_pixels = conv_pixels
        self._nx, self._ny = np.shape(conv_pixels)
        if compute_pixels is None:
            compute_pixels = copy.deepcopy(conv_pixels)
        assert np.shape(conv_pixels) == np.shape(compute_pixels)
        self._mask = compute_pixels
        self._partialInput = PartialImage(partial_read_bools=conv_pixels)
        self._partialOutput = PartialImage(partial_read_bools=compute_pixels)
        index_array_out = self._partialOutput.index_array
        index_array_input = self._partialInput.index_array
        kernel_shape = kernel.shape
        self.kernel_max_size = kernel_shape[0] * kernel_shape[1]

        image_index = 0
        self._image_frame_indexes = np.zeros((self._partialInput.num_partial, self.kernel_max_size), dtype='int')
        self._image_frame_psfs = np.zeros((self._partialInput.num_partial, self.kernel_max_size))
        self._image_frame_lengths = np.zeros((self._partialInput.num_partial), dtype='int')
        for x in range(index_array_input.shape[0]):
            for y in range(index_array_input.shape[1]):
                if conv_pixels[x][y]:
                    image_frame_psfs, image_frame_indexes, frame_length = self.pre_compute_frame_kernel((x, y),
                                                                                                        self._kernel[:, :],
                                                                                                        compute_pixels,
                                                                                                        index_array_out)

                    self._image_frame_indexes[image_index, :] = image_frame_indexes
                    self._image_frame_psfs[image_index, :] = image_frame_psfs
                    self._image_frame_lengths[image_index] = frame_length
                    image_index += 1

    def convolve2d(self, image):
        """
        2d convolution

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """
        image_array_partial = self._partialInput.partial_array(image)
        conv_array = self._convolve_jit(image_array_partial, num_data=self._partialOutput.num_partial,
                                        image_frame_kernels=self._image_frame_psfs,
                                        image_frame_indexes=self._image_frame_indexes,
                                        image_frame_lengths=self._image_frame_lengths)
        conv_image = self._partialOutput.image_from_partial(conv_array)
        return conv_image

    @staticmethod
    @numba_util.jit(nopython=numba_util.nopython, cache=numba_util.cache, parallel=numba_util.parallel)
    def pre_compute_frame_kernel(image_index, kernel, mask, index_array):
        """

        :param image_index: (int, int) index of pixels
        :param kernel: kernel, 2d rectangular array
        :param mask: mask (size of full image)
        :return:
        frame_kernels: values of kernel
        frame_indexes: (int) 1d index corresponding to the pixel receiving the kernel value
        frame_counter: number of pixels with non-zero addition due to kernel convolution
        """
        i0, j0 = image_index
        kernel_shape = kernel.shape
        kx, ky = kernel_shape[0], kernel_shape[1]
        mask_shape = index_array.shape
        nx, ny = mask_shape[0], mask_shape[1]
        kx2 = int((kx - 1) / 2)
        ky2 = int((ky - 1) / 2)
        frame_counter = 0
        frame_kernels = np.zeros(kx*ky)
        frame_indexes = np.zeros(kx*ky)

        for i in range(kx):
            for j in range(ky):
                if mask[i, j]:
                    x = i0 + i - kx2
                    y = j0 + j - ky2
                    if 0 <= x < nx and 0 <= y < ny:
                        frame_indexes[frame_counter] = index_array[x, y]
                        frame_kernels[frame_counter] = kernel[i, j]
                        frame_counter += 1
        return frame_kernels, frame_indexes, frame_counter

    @staticmethod
    @numba_util.jit(nopython=numba_util.nopython, cache=numba_util.cache, parallel=numba_util.parallel)
    def _convolve_jit(image_array, num_data, image_frame_kernels, image_frame_indexes, image_frame_lengths):
        """

        :param image_array: selected subset of image in 1d array conventions
        :param num_data: number of 1d data that get convolved light and are output
        :param image_conv_indexes: indexes of image (in 1d convention) those pixels get convolved
        :param image_frame_kernels: list of indexes that have a response for certain pixel (as a list
        :param image_frame_lengths: length of image_frame_kernels
        :return:
        """
        conv_array = np.zeros(num_data)
        for image_index in range(len(image_array)):  # loop through pixels that are to be blurred
            value = image_array[image_index]  # value of pixel that gets blurred
            frame_length = image_frame_lengths[image_index]  # number of pixels that gets assigned a fraction of the convolution
            frame_indexes = image_frame_indexes[image_index]  # list of 1d indexes that get added flux from the blurred image
            frame_kernels = image_frame_kernels[image_index]  # values of kernel for each frame indexes
            for kernel_index in range(frame_length):  # loop through all pixels that are impacted by the kernel of the pixel being blurred
                vector_index = frame_indexes[kernel_index]  # 1d coordinate of pixel to be added value
                kernel = frame_kernels[kernel_index]  # kernel response of pixel
                conv_array[vector_index] += value * kernel  # ad value to pixel
        return conv_array

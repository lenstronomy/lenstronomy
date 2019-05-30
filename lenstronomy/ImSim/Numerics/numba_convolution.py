import numpy as np

from lenstronomy.Util import numba_util
from lenstronomy.ImSim.Numerics.partial_image import PartialImage
from lenstronomy.Util import image_util


class NumbaConvolution(object):
    """
    class to convolve explicit pixels only

    the convolution is inspired by pyautolens: https://github.com/Jammy2211/PyAutoLens
    """
    def __init__(self, kernel, conv_pixels, compute_pixels=None, nopython=True, cache=True, parallel=False, memory_raise=True):
        """

        :param kernel: convolution kernel in units of the image pixels provided, odd length per axis
        :param conv_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        :param nopython: bool, numba jit setting to use python or compiled.
        :param cache: bool, numba jit setting to use cache
        :param parallel: bool, numba jit setting to use parallel mode
        :param memory_raise: bool, if True, checks whether memory required to store the convolution kernel is within certain bounds
        """
        #numba_util.nopython = nopython
        #numba_util.cache = cache
        #numba_util.parallel = parallel
        self._memory_raise = memory_raise
        self._kernel = kernel
        self._conv_pixels = conv_pixels
        self._nx, self._ny = np.shape(conv_pixels)
        if compute_pixels is None:
            compute_pixels = np.ones_like(conv_pixels)
            compute_pixels = np.array(compute_pixels, dtype=bool)
        assert np.shape(conv_pixels) == np.shape(compute_pixels)
        self._mask = compute_pixels
        self._partialInput = PartialImage(partial_read_bools=conv_pixels)
        self._partialOutput = PartialImage(partial_read_bools=compute_pixels)
        index_array_out = self._partialOutput.index_array
        index_array_input = self._partialInput.index_array
        kernel_shape = kernel.shape
        self.kernel_max_size = kernel_shape[0] * kernel_shape[1]

        image_index = 0
        if self._partialInput.num_partial * self.kernel_max_size > 10 ** 9 and self._memory_raise is True:
            raise ValueError("kernel length %s combined with data size %s requires %s memory elements, which might"
                             "exceed the memory limit and thus gives a raise. If you wish to ignore this raise, set"
                             " memory_raise=False" % (self.kernel_max_size, self._partialInput.num_partial, self._partialInput.num_partial * self.kernel_max_size))
        self._image_frame_indexes = np.zeros((self._partialInput.num_partial, self.kernel_max_size), dtype='int')
        self._image_frame_psfs = np.zeros((self._partialInput.num_partial, self.kernel_max_size))
        self._image_frame_lengths = np.zeros((self._partialInput.num_partial), dtype='int')
        for x in range(index_array_input.shape[0]):
            for y in range(index_array_input.shape[1]):
                if conv_pixels[x][y]:
                    image_frame_psfs, image_frame_indexes, frame_length = self._pre_compute_frame_kernel((x, y),
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
    @numba_util.jit()
    def _pre_compute_frame_kernel(image_index, kernel, mask, index_array):
        """

        :param image_index: (int, int) index of pixels
        :param kernel: kernel, 2d rectangular array
        :param mask: mask (size of full image)
        :return:
        frame_kernels: values of kernel
        frame_indexes: (int) 1d index corresponding to the pixel receiving the kernel value
        frame_counter: number of pixels with non-zero addition due to kernel convolution
        """
        kernel_shape = kernel.shape
        i0, j0 = image_index
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
                x = i0 + i - kx2
                y = j0 + j - ky2
                if 0 <= x < nx and 0 <= y < ny:
                    if mask[x, y]:
                        frame_indexes[frame_counter] = index_array[x, y]
                        frame_kernels[frame_counter] = kernel[i, j]
                        frame_counter += 1
        return frame_kernels, frame_indexes, frame_counter

    @staticmethod
    @numba_util.jit()
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


class SubgridNumbaConvolution(object):
    """
    class that inputs a supersampled grid and convolution kernel and computes the response on the regular grid
    This makes use of the regualr NumbaConvolution class as a loop through the different sub-pixel positions
    """

    def __init__(self, kernel_super, supersampling_factor, conv_pixels, compute_pixels=None, kernel_size=None, nopython=True, cache=True, parallel=False):
        """

        :param kernel_super: convolution kernel in units of super sampled pixels provided, odd length per axis
        :param supersampling_factor: factor of supersampling relative to pixel grid
        :param conv_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        :param nopython: bool, numba jit setting to use python or compiled.
        :param cache: bool, numba jit setting to use cache
        :param parallel: bool, numba jit setting to use parallel mode
        """
        self._nx, self._ny = conv_pixels.shape
        self._supersampling_factor = supersampling_factor
        # loop through the different supersampling sectors
        self._numba_conv_list = []
        if compute_pixels is None:
            compute_pixels = np.ones_like(conv_pixels)
            compute_pixels = np.array(compute_pixels, dtype=bool)

        for i in range(supersampling_factor):
            for j in range(supersampling_factor):
                # compute shifted psf kernel
                kernel = self._partial_kernel(kernel_super, i, j)
                if kernel_size is not None:
                    kernel = image_util.cut_edges(kernel, kernel_size)
                numba_conv = NumbaConvolution(kernel, conv_pixels, compute_pixels=compute_pixels, nopython=nopython, cache=cache, parallel=parallel)
                self._numba_conv_list.append(numba_conv)

    def convolve2d(self, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved and re-bined to regular resolution
        :return: convolved and re-bind image
        """
        conv_image = np.zeros((self._nx, self._ny))
        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                image_select = self._partial_image(image_high_res, i, j)
                conv_image += self._numba_conv_list[count].convolve2d(image_select)
                count += 1
        return conv_image

    def _partial_image(self, image_high_res, i, j):
        """

        :param image_high_res: 2d array supersampled
        :param i: index of super-sampled position in first axis
        :param j: index of super-sampled position in second axis
        :return: 2d array only selected the specific supersampled position within a regular pixel
        """
        return image_high_res[i::self._supersampling_factor, j::self._supersampling_factor]

    def _partial_kernel(self, kernel_super, i, j):
        """

        :param kernel_super: supersampled kernel
        :param i: index of super-sampled position in first axis
        :param j: index of super-sampled position in second axis
        :return: effective kernel rebinned to regular grid resulting from the subpersampled position (i,j)
        """
        n = len(kernel_super)
        kernel_size = int(round(n / float(self._supersampling_factor) + 1.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        n_match = kernel_size * self._supersampling_factor
        kernel_super_match = np.zeros((n_match, n_match))
        delta = int((n_match - n - self._supersampling_factor) / 2) + 1
        i0 = delta  # index where to start kernel for i=0
        j0 = delta  # index where to start kernel for j=0  (should be symmetric)
        kernel_super_match[i0 + i:i0 + i + n, j0 + j:j0 + j + n] = kernel_super
        #kernel_super_match = image_util.cut_edges(kernel_super_match, numPix=n)
        kernel = image_util.re_size(kernel_super_match, factor=self._supersampling_factor)
        return kernel

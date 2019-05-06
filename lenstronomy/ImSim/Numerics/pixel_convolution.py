import numpy as np

from lenstronomy.Util import numba_util
from lenstronomy.ImSim.Numerics.partial_image import PartialImage


class PartialConvolution(object):
    """
    class to convolve explicit pixels only
    """
    def __init__(self, kernel, conv_pixels, mask=None):
        """

        :param kernel: convolution kernel in units of the image pixels provided, odd length per axis
        :param conv_pixels: bool array same size as data, pixels to be convolved set to 1
        :param mask: bool array of size of image, these pixels (False) will not add blurring from other pixels
        """
        self._kernel = kernel
        self._conv_pixels = conv_pixels
        if mask is None:
            mask = np.ones_like(conv_pixels)
        assert np.shape(conv_pixels) == np.shape(mask)
        self._mask = mask
        self._partialInput = PartialImage(partial_read_bools=conv_pixels)
        self._partialOutput = PartialImage(partial_read_bools=mask)

    def convolve2d(self, image):
        """
        2d convolution

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """
        image_array_partial = self._partialInput.partial_array(image)
        conv_array = self._convolve_jit(image_array_partial, num_data=self._partialOutput.num_partial)
        conv_image = self._partialOutput.image_from_partial(conv_array)
        return conv_image


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
        for image_index in range(image_array):  # loop through pixels that are to be blurred
            value = image_array[image_index]  # value of pixel that gets blurred
            frame_length = image_frame_lengths[image_index]  # number of pixels that gets assigned a fraction of the convolution
            frame_indexes = image_frame_indexes[image_index]  # list of 1d indexes that get added flux from the blurred image
            frame_kernels = image_frame_kernels[image_index]  # values of kernel for eack frame indexes
            for kernel_index in range(frame_length):  # loop through all pixels that are impacted by the kernel of the pixel being blurred
                vector_index = frame_indexes[kernel_index]  # 1d coordinate of pixel to be added value
                kernel = frame_kernels[kernel_index]  # kernel response of pixel
                conv_array[vector_index] += value * kernel  # ad value to pixel
        return conv_array
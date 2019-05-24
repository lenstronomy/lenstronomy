import scipy.signal as signal
import scipy.ndimage as ndimage
import numpy as np

import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util


class PixelKernelConvolution(object):
    """
    class to compute convolutions for a given pixelized kernel (fft, grid)
    """
    def __init__(self, kernel, convolution_type='fft'):
        """

        :param kernel: 2d array, convolution kernel
        """
        self._kernel = kernel
        if convolution_type not in ['fft', 'grid']:
            raise ValueError('convolution_type %s not supported!' % convolution_type)
        self._type = convolution_type

    def convolution2d(self, image):
        """

        :param image: 2d array (image) to be convolved
        :return: fft convolution
        """
        if self._type == 'fft':
            image_conv = signal.fftconvolve(image, self._kernel, mode='same')
        elif self._type == 'grid':
            image_conv = signal.convolve2d(image, self._kernel, mode='same')
        else:
            raise ValueError('convolution_type %s not supported!' % self._type)
        return image_conv

    def re_size_convolve(self, image_low_res, image_high_res=None):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res)


class SubgridKernelConvolution(object):
    """
    class to compute the convolution on a supersampled grid with partial convolution computed on the regular grid
    """
    def __init__(self, kernel_supersampled, supersampling_factor, supersampling_kernel_size=None, convolution_type='fft'):
        """

        :param kernel_supersampled: kernel in supersampled pixels
        :param supersampling_factor: supersampling factor relative to the image pixel grid
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
        supersampled kernel
        """
        n_high = len(kernel_supersampled)
        self._supersampling_factor = supersampling_factor
        numPix = int(n_high / self._supersampling_factor)
        #if self._supersampling_factor % 2 == 0:
        #    self._kernel = kernel_util.averaging_even_kernel(kernel_supersampled, self._supersampling_factor)
        #else:
        #    self._kernel = util.averaging(kernel_supersampled, numGrid=n_high, numPix=numPix)
        if supersampling_kernel_size is None:
            kernel_low_res, kernel_high_res = np.zeros((3, 3)), kernel_supersampled
            self._low_res_convolution = False
        else:
            kernel_low_res, kernel_high_res = kernel_util.split_kernel(kernel_supersampled, supersampling_kernel_size,
                                                                       self._supersampling_factor)
            self._low_res_convolution = True
        self._low_res_conv = PixelKernelConvolution(kernel_low_res, convolution_type=convolution_type)
        self._high_res_conv = PixelKernelConvolution(kernel_high_res, convolution_type=convolution_type)

    def convolution2d(self, image):
        """

        :param image: 2d array (high resoluton image) to be convolved and re-sized
        :return: convolved image
        """

        image_high_res_conv = self._high_res_conv.convolution2d(image)
        image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        if self._low_res_convolution is True:
            image_resized = image_util.re_size(image, self._supersampling_factor)
            image_resized_conv += self._low_res_conv.convolution2d(image_resized)
        return image_resized_conv

    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_high_res_conv = self._high_res_conv.convolution2d(image_high_res)
        image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        if self._low_res_convolution is True:
            image_resized_conv += self._low_res_conv.convolution2d(image_low_res)
        return image_resized_conv


class MultiGaussianConvolution(object):
    """
    class to perform a convolution consisting of multiple 2d Gaussians
    This is aimed to lead to a speed-up without significant loss of accuracy do to the simplified convolution kernel
    relative to a pixelized kernel.
    """

    def __init__(self, sigma_list, fraction_list, pixel_scale, supersampling_factor=1, supersampling_convolution=False,
                 truncation=2):
        """

        :param sigma_list: list of std value of Gaussian kernel
        :param fraction_list: fraction of flux to be convoled with each Gaussian kernel
        :param pixel_scale: scale of pixel width (to convert sigmas into units of pixels)
        :param truncation: float. Truncate the filter at this many standard deviations.
        Default is 4.0.
        """
        self._num_gaussians = len(sigma_list)
        self._sigmas_scaled = np.array(sigma_list) / pixel_scale
        if supersampling_convolution is True:
            self._sigmas_scaled *= supersampling_factor
        self._fraction_list = fraction_list / np.sum(fraction_list)
        assert len(self._sigmas_scaled) == len(self._fraction_list)
        self._truncation = truncation
        self._pixel_scale = pixel_scale
        self._supersampling_factor = supersampling_factor
        self._supersampling_convolution = supersampling_convolution

    def convolution2d(self, image):
        """
        2d convolution

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """
        image_conv = None
        for i in range(self._num_gaussians):
            if image_conv is None:
                image_conv = ndimage.filters.gaussian_filter(image, self._sigmas_scaled[i], mode='nearest',
                                                             truncate=self._truncation) * self._fraction_list[i]
            else:
                image_conv += ndimage.filters.gaussian_filter(image, self._sigmas_scaled[i], mode='nearest',
                                                              truncate=self._truncation) * self._fraction_list[i]
        return image_conv

    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        if self._supersampling_convolution is True:
            image_high_res_conv = self.convolution2d(image_high_res)
            image_resized_conv = image_util.re_size(image_high_res_conv, self._supersampling_factor)
        else:
            image_resized_conv = self.convolution2d(image_low_res)
        return image_resized_conv

    def pixel_kernel(self, num_pix):
        """
        computes a pixelized kernel from the MGE parameters

        :param num_pix: int, size of kernel (odd number per axis)
        :return: pixel kernel centered
        """
        from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
        mg = MultiGaussian()
        x, y = util.make_grid(numPix=num_pix, deltapix=self._pixel_scale)
        kernel = mg.function(x, y, amp=self._fraction_list, sigma=self._sigmas_scaled)
        kernel = util.array2image(kernel)
        return kernel / np.sum(kernel)


class FWHMGaussianConvolution(object):
    """
    uses a two-dimensional Gaussian function with same FWHM of given kernel as approximation
    """
    def __init__(self, kernel, truncation=4):
        """

        :param kernel: 2d kernel
        :param truncation: sigma scaling of kernel truncation
        """
        fwhm = kernel_util.fwhm_kernel(kernel)
        self._sigma = util.fwhm2sigma(fwhm)
        self._truncation = truncation

    def convolution2d(self, image):
        """
        2d convolution

        :param image: 2d numpy array, image to be convolved
        :return: convolved image, 2d numpy array
        """

        image_conv = ndimage.filters.gaussian_filter(image, self._sigma, mode='nearest', truncate=self._truncation)
        return image_conv


class MGEConvolution(object):
    """
    approximates a 2d kernel with an azimuthal Multi-Gaussian expansion
    """
    def __init__(self, kernel, pixel_scale, order=1):
        """

        :param kernel: 2d convolution kernel (centered, odd axis number)
        :param order: order of Multi-Gaussian Expansion
        """
        #kernel_util.fwhm_kernel(kernel)
        amps, sigmas, norm = kernel_util.mge_kernel(kernel, order=order)
        # make instance o MultiGaussian convolution kernel
        self._mge_conv = MultiGaussianConvolution(sigma_list=sigmas*pixel_scale, fraction_list=np.array(amps) / np.sum(amps),
                                                  pixel_scale=pixel_scale, truncation=4)
        self._kernel = kernel
        # store difference between MGE approximation and real kernel

    def convolution2d(self, image):
        """

        :param image:
        :return:
        """
        return self._mge_conv.convolution2d(image)

    def kernel_difference(self):
        """

        :return: difference between true kernel and MGE approximation
        """
        kernel_mge = self._mge_conv.pixel_kernel(num_pix=len(self._kernel))
        return self._kernel - kernel_mge






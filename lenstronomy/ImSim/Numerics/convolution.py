from scipy import fftpack, ndimage, signal
import numpy as np
import threading
#from scipy._lib._version import NumpyVersion
_rfft_mt_safe = True  # (NumpyVersion(np.__version__) >= '1.9.0.dev-e24486e')
_rfft_lock = threading.Lock()

import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


@export
class PixelKernelConvolution(object):
    """
    class to compute convolutions for a given pixelized kernel (fft, grid)
    """
    def __init__(self, kernel, convolution_type='fft_static'):
        """

        :param kernel: 2d array, convolution kernel
        :param convolution_type: string, 'fft', 'grid', 'fft_static' mode of 2d convolution
        """
        self._kernel = kernel
        if convolution_type not in ['fft', 'grid', 'fft_static']:
            raise ValueError('convolution_type %s not supported!' % convolution_type)
        self._type = convolution_type
        self._pre_computed = False

    def pixel_kernel(self, num_pix=None):
        """
        access pixelated kernel

        :param num_pix: size of returned kernel (odd number per axis). If None, return the original kernel.
        :return: pixel kernel centered
        """
        if num_pix is not None:
            return kernel_util.cut_psf(self._kernel, num_pix)
        return self._kernel

    def copy_transpose(self):
        """
        
        :return: copy of the class with kernel set to the transpose of original one
        """
        return PixelKernelConvolution(self._kernel.T, convolution_type=self._type)

    def convolution2d(self, image):
        """

        :param image: 2d array (image) to be convolved
        :return: fft convolution
        """
        if self._type == 'fft':
            image_conv = signal.fftconvolve(image, self._kernel, mode='same')
        elif self._type == 'fft_static':
            image_conv = self._static_fft(image, mode='same')
        elif self._type == 'grid':
            image_conv = signal.convolve2d(image, self._kernel, mode='same')
        else:
            raise ValueError('convolution_type %s not supported!' % self._type)
        return image_conv

    def _static_fft(self, image, mode='same'):
        """
        scipy fft convolution with saved static fft kernel

        :param image: 2d numpy array to be convolved
        :return:
        """
        in1 = image
        in1 = np.asarray(in1)
        if self._pre_computed is False:
            self._s1, self._s2, self._complex_result, self._shape, self._fshape, self._fslice, self._sp2 = self._static_pre_compute(image)
            self._pre_computed = True
        s1, s2, complex_result, shape, fshape, fslice, sp2 = self._s1, self._s2, self._complex_result, self._shape, self._fshape, self._fslice, self._sp2
        #if in1.ndim == in2.ndim == 0:  # scalar inputs
        #    return in1 * in2
        #elif not in1.ndim == in2.ndim:
        #    raise ValueError("in1 and in2 should have the same dimensionality")
        #elif in1.size == 0 or in2.size == 0:  # empty arrays
        #    return np.array([])


        # Check that input sizes are compatible with 'valid' mode
        #if _inputs_swap_needed(mode, s1, s2):
            # Convolution is commutative; order doesn't have any effect on output
            # only applicable for 'valid' mode
        #    in1, s1, in2, s2 = in2, s2, in1, s1

        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.
        if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
            try:
                sp1 = np.fft.rfftn(in1, fshape)
                ret = (np.fft.irfftn(sp1 * sp2, fshape)[fslice].copy())
            finally:
                if not _rfft_mt_safe:
                    _rfft_lock.release()
        else:
            # If we're here, it's either because we need a complex result, or we
            # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
            # is already in use by another thread).  In either case, use the
            # (threadsafe but slower) SciPy complex-FFT routines instead.
            sp1 = fftpack.fftn(in1, fshape)
            ret = fftpack.ifftn(sp1 * sp2)[fslice].copy()
            if not complex_result:
                ret = ret.real

        if mode == "full":
            return ret
        elif mode == "same":
            return _centered(ret, s1)
        elif mode == "valid":
            return _centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                             " 'same', or 'full'.")

    def _static_pre_compute(self, image):
        """
        pre-compute Fourier transformed kernel and shape quantities to speed up convolution

        :param image: 2d numpy array
        :return:
        """
        in1 = image
        in2 = self._kernel
        s1 = np.array(in1.shape)
        s2 = np.array(in2.shape)
        complex_result = (np.issubdtype(in1.dtype, np.complexfloating) or
                          np.issubdtype(in2.dtype, np.complexfloating))
        shape = s1 + s2 - 1

        # Check that input sizes are compatible with 'valid' mode
        # if _inputs_swap_needed(mode, s1, s2):
        # Convolution is commutative; order doesn't have any effect on output
        # only applicable for 'valid' mode
        #    in1, s1, in2, s2 = in2, s2, in1, s1

        # Speed up FFT by padding to optimal size for FFTPACK
        fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
        fslice = tuple([slice(0, int(sz)) for sz in shape])
        # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
        # sure we only call rfftn/irfftn from one thread at a time.
        if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
            try:
                sp2 = np.fft.rfftn(in2, fshape)
            finally:
                if not _rfft_mt_safe:
                    _rfft_lock.release()
        else:
            # If we're here, it's either because we need a complex result, or we
            # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
            # is already in use by another thread).  In either case, use the
            # (threadsafe but slower) SciPy complex-FFT routines instead.
            sp2 = fftpack.fftn(in2, fshape)
        return s1, s2, complex_result, shape, fshape, fslice, sp2

    def re_size_convolve(self, image_low_res, image_high_res=None):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res)


@export
class SubgridKernelConvolution(object):
    """
    class to compute the convolution on a supersampled grid with partial convolution computed on the regular grid
    """
    def __init__(self, kernel_supersampled, supersampling_factor, supersampling_kernel_size=None, convolution_type='fft_static'):
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


@export
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


@export
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


@export
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

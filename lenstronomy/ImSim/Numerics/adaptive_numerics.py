from lenstronomy.ImSim.Numerics.numba_convolution import SubgridNumbaConvolution, NumbaConvolution
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.Util import kernel_util
from lenstronomy.Util import image_util


class AdaptiveConvolution(object):
    """
    This class performs convolutions of a subset of pixels at higher supersampled resolution
    Goal: speed up relative to higher resolution FFT when only considereing a (small) subset of pixels to be convolved
    on the higher resolution grid.

    strategy:
    1. lower resolution convolution over full image with FFT
    2. subset of pixels with higher resolution Numba convolution (with smaller kernel)
    3. the same subset of pixels with low resolution Numba convolution (with same kernel as step 2)
    adaptive solution is 1 + 2 -3

    """
    def __init__(self, kernel_super, supersampling_factor, conv_supersample_pixels, supersampling_kernel_size=None,
                 compute_pixels=None, nopython=True, cache=True, parallel=False):
        """

        :param kernel_super: convolution kernel in units of super sampled pixels provided, odd length per axis
        :param supersampling_factor: factor of supersampling relative to pixel grid
        :param conv_supersample_pixels: bool array same size as data, pixels to be convolved and their light to be blurred
        :param supersampling_kernel_size: number of pixels (in units of the image pixels) that are convolved with the
        supersampled kernel
        :param compute_pixels: bool array of size of image, these pixels (if True) will get blurred light from other pixels
        :param nopython: bool, numba jit setting to use python or compiled.
        :param cache: bool, numba jit setting to use cache
        :param parallel: bool, numba jit setting to use parallel mode
        """
        kernel = kernel_util.degrade_kernel(kernel_super, degrading_factor=supersampling_factor)
        self._low_res_conv = PixelKernelConvolution(kernel, convolution_type='fft')
        if supersampling_kernel_size is None:
            supersampling_kernel_size = len(kernel)

        n_cut_super = supersampling_kernel_size * supersampling_factor
        if n_cut_super % 2 == 0:
            n_cut_super += 1
        #kernel_super_cut = image_util.cut_edges(kernel_super, n_cut_super)
        #kernel_cut = kernel_util.degrade_kernel(kernel_super_cut, degrading_factor=supersampling_factor)
        kernel_super_cut = image_util.cut_edges(kernel_super, n_cut_super)
        kernel_cut = kernel_util.degrade_kernel(kernel_super_cut, degrading_factor=supersampling_factor)

        self._low_res_partial = NumbaConvolution(kernel_cut, conv_supersample_pixels, compute_pixels=compute_pixels,
                                                 nopython=nopython, cache=cache, parallel=parallel, memory_raise=True)
        self._hig_res_partial = SubgridNumbaConvolution(kernel_super_cut, supersampling_factor, conv_supersample_pixels,
                                                        compute_pixels=compute_pixels, nopython=nopython, cache=cache,
                                                        parallel=parallel)#, kernel_size=len(kernel_cut))
        self._supersampling_factor = supersampling_factor

    def re_size_convolve(self, image_low_res, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_low_res_conv = self._low_res_conv.convolution2d(image_low_res)
        image_low_res_partial_conv = self._low_res_partial.convolve2d(image_low_res)
        image_high_res_partial_conv = self._hig_res_partial.convolve2d(image_high_res)
        return image_low_res_conv + image_high_res_partial_conv - image_low_res_partial_conv

    def convolve2d(self, image_high_res):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :return: convolved and re-sized image
        """
        image_low_res = image_util.re_size(image_high_res, factor=self._supersampling_factor)
        return self.re_size_convolve(image_low_res, image_high_res)

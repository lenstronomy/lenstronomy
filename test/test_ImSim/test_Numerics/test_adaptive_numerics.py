__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.Numerics.adaptive_numerics import AdaptiveConvolution
from lenstronomy.ImSim.Numerics.convolution import SubgridKernelConvolution
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as util
import pytest


class TestAdaptiveConvolution(object):

    def setup(self):
        self.supersampling_factor = 3
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.delta_pix = 1.
        x, y = util.make_grid(20, deltapix=self.delta_pix)
        x_sub, y_sub = util.make_grid(20*self.supersampling_factor, deltapix=self.delta_pix/self.supersampling_factor)
        kwargs = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)
        flux_sub = lightModel.surface_brightness(x_sub, y_sub, kwargs)
        self.model_sub = util.array2image(flux_sub)

        x, y = util.make_grid(5, deltapix=self.delta_pix)
        kwargs_kernel = [{'amp': 1, 'sigma_x': 1, 'sigma_y': 1, 'center_x': 0, 'center_y': 0}]
        kernel = lightModel.surface_brightness(x, y, kwargs_kernel)
        self.kernel = util.array2image(kernel) / np.sum(kernel)

        x_sub, y_sub = util.make_grid(5*self.supersampling_factor, deltapix=self.delta_pix/self.supersampling_factor)
        kernel_sub = lightModel.surface_brightness(x_sub, y_sub, kwargs_kernel)
        self.kernel_sub = util.array2image(kernel_sub) / np.sum(kernel_sub)

    def test_convolve2d(self):
        #kernel_supersampled = kernel_util.subgrid_kernel(self.kernel, self.supersampling_factor, odd=True, num_iter=5)
        subgrid_conv = SubgridKernelConvolution(self.kernel_sub, self.supersampling_factor, supersampling_kernel_size=None, convolution_type='fft')
        model_subgrid_conv = subgrid_conv.convolution2d(self.model_sub)

        conv_supersample_pixels = np.zeros_like(self.model)
        conv_supersample_pixels = np.array(conv_supersample_pixels, dtype=bool)
        conv_supersample_pixels[self.model > np.max(self.model)/20] = True
        adaptive_conv = AdaptiveConvolution(self.kernel_sub, self.supersampling_factor, conv_supersample_pixels, supersampling_kernel_size=5, compute_pixels=None)

        model_adaptive_conv = adaptive_conv.convolve2d(self.model_sub)
        npt.assert_almost_equal(np.sum(model_subgrid_conv), np.sum(model_adaptive_conv), decimal=2)
        npt.assert_almost_equal(model_subgrid_conv, model_adaptive_conv, decimal=2)

        conv_supersample_pixels = np.zeros_like(self.model)
        conv_supersample_pixels = np.array(conv_supersample_pixels, dtype=bool)
        conv_supersample_pixels[self.model > np.max(self.model) / 2] = True
        adaptive_conv = AdaptiveConvolution(self.kernel_sub, self.supersampling_factor, conv_supersample_pixels,
                                            supersampling_kernel_size=1, compute_pixels=None)

        model_adaptive_conv = adaptive_conv.convolve2d(self.model_sub)

        npt.assert_almost_equal(np.sum(model_subgrid_conv), np.sum(model_adaptive_conv), decimal=2)
        npt.assert_almost_equal(model_subgrid_conv, model_adaptive_conv, decimal=2)


if __name__ == '__main__':
    pytest.main()

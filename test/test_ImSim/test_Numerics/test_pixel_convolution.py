__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.Numerics.numba_convolution import NumbaConvolution, SubgridNumbaConvolution
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import util
from lenstronomy.Util import image_util
import pytest


class TestPixelConvolution(object):

    def setup(self):
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.delta_pix = 1
        self.num_pix = 10
        self.num_pix_kernel = 7
        x, y = util.make_grid(numPix=self.num_pix_kernel, deltapix=self.delta_pix)
        kwargs_kernel = [{'amp': 1, 'sigma_x': 3, 'sigma_y': 3, 'center_x': 0, 'center_y': 0}]
        kernel = lightModel.surface_brightness(x, y, kwargs_kernel)
        self.kernel = util.array2image(kernel)
        self.kernel /= np.sum(self.kernel)

        x, y = util.make_grid(numPix=self.num_pix, deltapix=self.delta_pix)
        kwargs = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)

    def test_convolve2d(self):
        conv_pixels = np.ones_like(self.model)
        conv_pixels = np.array(conv_pixels, dtype=bool)
        numba_conv = NumbaConvolution(kernel=self.kernel, conv_pixels=conv_pixels, compute_pixels=conv_pixels)

        model_conv_numba = numba_conv.convolve2d(self.model)

        pixel_conv = PixelKernelConvolution(kernel=self.kernel)
        image_convolved = pixel_conv.convolution2d(self.model)
        npt.assert_almost_equal(model_conv_numba, image_convolved, decimal=10)


class TestSubgirdNumbaConvolution(object):
    def setup(self):
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.supersampling_factor = 3
        self.delta_pix = 1
        self.num_pix = 10
        self.num_pix_kernel = 7
        x, y = util.make_grid(numPix=self.num_pix_kernel, deltapix=self.delta_pix)
        kwargs_kernel = [{'amp': 1, 'sigma_x': 3, 'sigma_y': 3, 'center_x': 0, 'center_y': 0}]
        kernel = lightModel.surface_brightness(x, y, kwargs_kernel)
        self.kernel = util.array2image(kernel)
        self.kernel /= np.sum(self.kernel)

        x_sub, y_sub = util.make_grid(numPix=self.num_pix_kernel, deltapix=self.delta_pix, subgrid_res=self.supersampling_factor)
        kernel_super = lightModel.surface_brightness(x_sub, y_sub, kwargs_kernel)
        self.kernel_super = util.array2image(kernel_super)
        self.kernel_super /= np.sum(self.kernel_super)

        x_sub, y_sub = util.make_grid(numPix=self.num_pix, deltapix=self.delta_pix, subgrid_res=self.supersampling_factor)
        kwargs = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x_sub, y_sub, kwargs)
        self.model_super = util.array2image(flux)
        self.model = image_util.re_size(self.model_super, factor=self.supersampling_factor)

    def test_convolve2d(self):
        conv_pixels = np.ones_like(self.model)
        conv_pixels = np.array(conv_pixels, dtype=bool)
        numba_conv = SubgridNumbaConvolution(kernel_super=self.kernel_super, conv_pixels=conv_pixels,
                                             compute_pixels=conv_pixels, supersampling_factor=self.supersampling_factor)

        model_conv_numba = numba_conv.convolve2d(self.model_super)
        pixel_conv = PixelKernelConvolution(kernel=self.kernel_super)
        image_convolved = pixel_conv.convolution2d(self.model_super)
        image_convolved = image_util.re_size(image_convolved, factor=self.supersampling_factor)
        npt.assert_almost_equal(model_conv_numba, image_convolved, decimal=10)


if __name__ == '__main__':
    pytest.main()

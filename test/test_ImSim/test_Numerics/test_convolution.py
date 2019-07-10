__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim.Numerics.convolution import MultiGaussianConvolution, PixelKernelConvolution, \
    SubgridKernelConvolution, MGEConvolution
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.util as util
import pytest


class TestPixelKernelConvolution(object):

    def setup(self):
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.delta_pix = 1
        x, y = util.make_grid(10, deltapix=self.delta_pix)
        kwargs = [{'amp': 1, 'sigma_x': 1, 'sigma_y': 1, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)

    def test_convolve2d(self):
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1
        pixel_conv = PixelKernelConvolution(kernel=kernel)
        image_convolved = pixel_conv.convolution2d(self.model)
        npt.assert_almost_equal(np.sum(image_convolved), np.sum(self.model), decimal=2)


class TestSubgridKernelConvolution(object):

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

    def test_fft_scipy_static(self):

        supersampling_factor = 2
        conv_sicpy = SubgridKernelConvolution(self.kernel, supersampling_factor, supersampling_kernel_size=None,
                                        convolution_type='fft')

        conv_static = SubgridKernelConvolution(self.kernel, supersampling_factor, supersampling_kernel_size=None,
                                                  convolution_type='fft_static')

        model_conv_scipy = conv_sicpy.convolution2d(self.model)
        model_conv_static = conv_static.convolution2d(self.model)
        npt.assert_almost_equal(model_conv_static, model_conv_scipy, decimal=3)

    def test_convolve2d(self):
        #kernel_supersampled = kernel_util.subgrid_kernel(self.kernel, self.supersampling_factor, odd=True, num_iter=5)
        subgrid_conv = SubgridKernelConvolution(self.kernel_sub, self.supersampling_factor, supersampling_kernel_size=None, convolution_type='fft')
        model_subgrid_conv = subgrid_conv.convolution2d(self.model_sub)

        supersampling_factor = 1
        conv = SubgridKernelConvolution(self.kernel, supersampling_factor, supersampling_kernel_size=None,
                                        convolution_type='fft')
        model_conv = conv.convolution2d(self.model)
        npt.assert_almost_equal(np.sum(model_subgrid_conv), np.sum(model_conv), decimal=1)
        npt.assert_almost_equal(model_subgrid_conv, model_conv, decimal=1)


        #kernel_supersampled = kernel_util.subgrid_kernel(self.kernel, self.supersampling_factor, odd=True, num_iter=5)
        subgrid_conv_split = SubgridKernelConvolution(self.kernel_sub, self.supersampling_factor, supersampling_kernel_size=5,
                                                convolution_type='fft')
        model_subgrid_conv_split = subgrid_conv_split.convolution2d(self.model_sub)
        npt.assert_almost_equal(np.sum(model_subgrid_conv), np.sum(model_subgrid_conv_split), decimal=8)
        npt.assert_almost_equal(model_subgrid_conv, model_subgrid_conv_split, decimal=8)

        subgrid_conv_split = SubgridKernelConvolution(self.kernel_sub, self.supersampling_factor, supersampling_kernel_size=3,
                                                      convolution_type='fft')
        model_subgrid_conv_split = subgrid_conv_split.convolution2d(self.model_sub)
        npt.assert_almost_equal(np.sum(model_subgrid_conv), np.sum(model_subgrid_conv_split), decimal=5)
        npt.assert_almost_equal(model_subgrid_conv, model_subgrid_conv_split, decimal=3)


class TestMultiGaussianConvolution(object):

    def setup(self):
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.delta_pix = 1
        x, y = util.make_grid(10, deltapix=self.delta_pix)
        kwargs = [{'amp': 1, 'sigma_x': 1, 'sigma_y': 1, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)

    def test_convolve2d(self):
        sigma_list = [0.5, 1, 2]
        fraction_list = [0.5, 0.2, 0.3]
        mge_conv = MultiGaussianConvolution(sigma_list=sigma_list, fraction_list=fraction_list, pixel_scale=self.delta_pix)
        image_convolved = mge_conv.convolution2d(self.model)
        npt.assert_almost_equal(np.sum(image_convolved), np.sum(self.model), decimal=2)


class TestMGEConvolution(object):

    def setup(self):
        lightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.delta_pix = 1
        x, y = util.make_grid(10, deltapix=self.delta_pix)
        kwargs = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        flux = lightModel.surface_brightness(x, y, kwargs)
        self.model = util.array2image(flux)

    def test_convolve2d(self):

        sigma_list = [2, 3, 4]
        fraction_list = [0.5, 0.2, 0.3]
        mg_conv = MultiGaussianConvolution(sigma_list=sigma_list, fraction_list=fraction_list, pixel_scale=self.delta_pix)
        pixel_kernel = mg_conv.pixel_kernel(num_pix=11)
        mge_conv = MGEConvolution(pixel_kernel, pixel_scale=self.delta_pix, order=20)
        image_conv_mg = mg_conv.convolution2d(self.model)
        image_conv_mge = mge_conv.convolution2d(self.model)
        npt.assert_almost_equal(image_conv_mge/np.max(image_conv_mg), image_conv_mg/np.max(image_conv_mg), decimal=2)

        diff_kernel = mge_conv.kernel_difference()
        npt.assert_almost_equal(diff_kernel, pixel_kernel - mge_conv._mge_conv.pixel_kernel(len(pixel_kernel)))


if __name__ == '__main__':
    pytest.main()

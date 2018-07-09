import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.psf import PSF
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util


class TestData(object):

    def setup(self):
        self.deltaPix = 0.05
        fwhm = 0.2
        kwargs_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncate': 5, 'pixel_size': self.deltaPix}
        self.psf_gaussian = PSF(kwargs_psf=kwargs_gaussian)
        kernel_point_source = kernel_util.kernel_gaussian(kernel_numPix=21, deltaPix=self.deltaPix, fwhm=fwhm)
        kwargs_pixel = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}
        self.psf_pixel = PSF(kwargs_psf=kwargs_pixel)

    def test_kernel_point_source(self):
        kernel_gaussian = self.psf_gaussian.kernel_point_source
        kernel_pixel = self.psf_pixel.kernel_point_source
        assert len(kernel_gaussian) == 21
        assert len(kernel_pixel) == 21

    def test_psf_convolution(self):

        deltaPix = 0.05
        fwhm = 0.2
        fwhm_object = 0.1
        kwargs_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncate': 5, 'pixel_size': deltaPix}
        psf_gaussian = PSF(kwargs_psf=kwargs_gaussian)
        kernel_point_source = kernel_util.kernel_gaussian(kernel_numPix=21, deltaPix=deltaPix, fwhm=fwhm)
        kwargs_pixel = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}
        psf_pixel = PSF(kwargs_psf=kwargs_pixel)

        subgrid_res_input = 15

        grid = kernel_util.kernel_gaussian(kernel_numPix=11*subgrid_res_input, deltaPix=deltaPix / float(subgrid_res_input), fwhm=fwhm_object)
        grid_true = image_util.re_size(grid, subgrid_res_input)
        grid_conv = psf_gaussian.psf_convolution(grid, deltaPix / float(subgrid_res_input))
        grid_conv_true = image_util.re_size(grid_conv, subgrid_res_input)

        # subgrid resoluton Gaussian convolution
        for i in range(1, subgrid_res_input+1):

            subgrid_res = i

            grid = kernel_util.kernel_gaussian(kernel_numPix=11*subgrid_res, deltaPix=deltaPix / float(subgrid_res), fwhm=fwhm_object)
            grid_conv = psf_gaussian.psf_convolution(grid, deltaPix / float(subgrid_res))
            grid_conv_finite = image_util.re_size(grid_conv, subgrid_res)
            min_diff = np.min(grid_conv_true-grid_conv_finite)
            max_diff = np.max(grid_conv_true-grid_conv_finite)
            print(min_diff, max_diff)
        npt.assert_almost_equal(min_diff, 0, decimal=7)
        npt.assert_almost_equal(max_diff, 0, decimal=7)
        # subgrid resoluton Pixel convolution
        for i in range(1,subgrid_res_input+1):

            subgrid_res = i

            grid = kernel_util.kernel_gaussian(kernel_numPix=11*subgrid_res, deltaPix=deltaPix / float(subgrid_res), fwhm=fwhm_object)
            grid_conv = psf_pixel.psf_convolution(grid, deltaPix / float(subgrid_res), subgrid_res=subgrid_res, psf_subgrid=True)
            grid_conv_finite = image_util.re_size(grid_conv, subgrid_res)
            min_diff = np.min(grid_conv_true-grid_conv_finite)
            max_diff = np.max(grid_conv_true-grid_conv_finite)
            print(min_diff, max_diff)
        npt.assert_almost_equal(min_diff, 0, decimal=3)
        npt.assert_almost_equal(max_diff, 0, decimal=3)
        # subgrid ray-tracing but pixel convolution on normal grid
        for i in range(1,subgrid_res_input+1):

            subgrid_res = i

            grid = kernel_util.kernel_gaussian(kernel_numPix=11*subgrid_res, deltaPix=deltaPix / float(subgrid_res), fwhm=0.2)
            grid_finite = image_util.re_size(grid, subgrid_res)
            grid_conv_finite = psf_pixel.psf_convolution(grid_finite, deltaPix, subgrid_res=1)
            min_diff = np.min(grid_conv_true-grid_conv_finite)
            max_diff = np.max(grid_conv_true-grid_conv_finite)
            print(min_diff, max_diff)
        npt.assert_almost_equal(min_diff, 0, decimal=3)
        npt.assert_almost_equal(max_diff, 0, decimal=3)

    def test_fwhm(self):
        deltaPix = 0.05
        fwhm = 0.1
        kwargs = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncate': 5, 'pixel_size': deltaPix}
        fwhm_compute = self.psf_gaussian.psf_fwhm(kwargs=kwargs, deltaPix=deltaPix)
        assert fwhm_compute == fwhm


if __name__ == '__main__':
    pytest.main()

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

    def test_kernel_subsampled(self):
        deltaPix = 0.05  # pixel size of image
        numPix = 40  # number of pixels per axis
        subsampling_res = 3  # subsampling scale factor (in each dimension)
        fwhm = 0.3  # FWHM of the PSF kernel
        fwhm_object = 0.2  # FWHM of the Gaussian source to be convolved

        # create Gaussian/Pixelized kernels
        # first we create the sub-sampled kernel
        kernel_point_source_subsampled = kernel_util.kernel_gaussian(kernel_numPix=11*subsampling_res, deltaPix=deltaPix/subsampling_res, fwhm=fwhm)
        # to have the same consistent kernel, we re-size (average over the sub-sampled pixels) the sub-sampled kernel
        kernel_point_source = image_util.re_size(kernel_point_source_subsampled, subsampling_res)
        # here we create the two PSF() classes
        kwargs_pixel_subsampled = {'psf_type': 'PIXEL', 'kernel_point_source_subsampled': kernel_point_source_subsampled, 'subsampling_factor': subsampling_res}
        psf_pixel_subsampled = PSF(kwargs_psf=kwargs_pixel_subsampled)
        kwargs_pixel = {'psf_type': 'PIXEL',
                        'kernel_point_source': kernel_point_source}
        psf_pixel = PSF(kwargs_psf=kwargs_pixel)

        # here we create the image of the Gaussian source and convolve it with the regular kernel
        image_unconvolved = kernel_util.kernel_gaussian(kernel_numPix=numPix, deltaPix=deltaPix, fwhm=fwhm_object)
        image_convolved_regular = psf_pixel.psf_convolution_new(image_unconvolved, subgrid_res=1, subsampling_size=None)

        # here we create the image by computing the sub-sampled Gaussian source and convolve it with the sub-sampled PSF kernel
        image_unconvolved_highres = kernel_util.kernel_gaussian(kernel_numPix=numPix*subsampling_res, deltaPix=deltaPix/subsampling_res, fwhm=fwhm_object) * subsampling_res**2
        image_convolved_subsampled = psf_pixel_subsampled.psf_convolution_new(image_unconvolved_highres, subgrid_res=subsampling_res, subsampling_size=5)

        # We demand the two procedures to be the same up to the numerics affecting the finite resolution
        npt.assert_almost_equal(np.sum(image_convolved_regular), np.sum(image_convolved_subsampled), decimal=8)
        npt.assert_almost_equal((image_convolved_subsampled - image_convolved_regular) / (np.max(image_convolved_subsampled)), 0, decimal=2)

    def test_fwhm(self):
        deltaPix = 0.05
        fwhm = 0.1
        kwargs = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncate': 5, 'pixel_size': deltaPix}
        fwhm_compute = self.psf_gaussian.psf_fwhm(kwargs=kwargs, deltaPix=deltaPix)
        assert fwhm_compute == fwhm

        kernel = kernel_util.kernel_gaussian(kernel_numPix=11, deltaPix=deltaPix, fwhm=fwhm)
        kwargs = {'psf_type': 'PIXEL',  'truncate': 5, 'pixel_size': deltaPix, 'kernel_point_source': kernel}
        fwhm_compute = self.psf_gaussian.psf_fwhm(kwargs=kwargs, deltaPix=deltaPix)
        npt.assert_almost_equal(fwhm_compute, fwhm, decimal=6)

        kwargs = {'psf_type': 'PIXEL', 'truncate': 5, 'pixel_size': deltaPix, 'kernel_point_source_subsampled': kernel}
        fwhm_compute = self.psf_gaussian.psf_fwhm(kwargs=kwargs, deltaPix=deltaPix)
        npt.assert_almost_equal(fwhm_compute, fwhm, decimal=6)


if __name__ == '__main__':
    pytest.main()

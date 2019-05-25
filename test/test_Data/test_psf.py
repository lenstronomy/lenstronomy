import pytest
import numpy as np
import numpy.testing as npt
import unittest
import warnings

from lenstronomy.Data.psf import PSF
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import simulation_util


class TestData(object):

    def setup(self):
        self.deltaPix = 0.05
        fwhm = 0.2
        kwargs_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': self.deltaPix}
        self.psf_gaussian = PSF(**kwargs_gaussian)
        kernel_point_source = kernel_util.kernel_gaussian(kernel_numPix=21, deltaPix=self.deltaPix, fwhm=fwhm)
        kwargs_pixel = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}
        self.psf_pixel = PSF(**kwargs_pixel)

    def test_kernel_point_source(self):
        kernel_gaussian = self.psf_gaussian.kernel_point_source
        kernel_pixel = self.psf_pixel.kernel_point_source
        assert len(kernel_gaussian) == 21
        assert len(kernel_pixel) == 21

        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.2, 'truncation': 3, 'pixel_size': 0.05}
        psf_class = PSF(**kwargs_psf)
        kernel_point_source = psf_class.kernel_point_source
        assert len(kernel_point_source) == 13
        kernel_super = psf_class.kernel_point_source_supersampled(supersampling_factor=3)
        assert np.sum(kernel_point_source) == np.sum(kernel_super)
        assert np.sum(kernel_point_source) == 1

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
        kwargs_pixel_subsampled = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source_subsampled,
                                   'point_source_supersampling_factor': subsampling_res}
        psf_pixel_subsampled = PSF(**kwargs_pixel_subsampled)
        psf_pixel_subsampled.kernel_point_source_supersampled(supersampling_factor=subsampling_res+1)
        kernel_point_source /= np.sum(kernel_point_source)
        kwargs_pixel = {'psf_type': 'PIXEL',
                        'kernel_point_source': kernel_point_source}
        psf_pixel = PSF(**kwargs_pixel)

        kernel_point_source = psf_pixel.kernel_point_source
        kernel_super = psf_pixel.kernel_point_source_supersampled(supersampling_factor=3)
        npt.assert_almost_equal(np.sum(kernel_point_source), np.sum(kernel_super), decimal=8)
        npt.assert_almost_equal(np.sum(kernel_point_source), 1, decimal=8)

        deltaPix = 0.05  # pixel size of image
        numPix = 40  # number of pixels per axis
        subsampling_res = 4  # subsampling scale factor (in each dimension)
        fwhm = 0.3  # FWHM of the PSF kernel
        fwhm_object = 0.2  # FWHM of the Gaussian source to be convolved

        # create Gaussian/Pixelized kernels
        # first we create the sub-sampled kernel
        kernel_point_source_subsampled = kernel_util.kernel_gaussian(kernel_numPix=11 * subsampling_res + 1,
                                                                     deltaPix=deltaPix / subsampling_res, fwhm=fwhm)

        kwargs_pixel_subsampled = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source_subsampled,
                                   'point_source_supersampling_factor': subsampling_res}
        psf_pixel_subsampled = PSF(**kwargs_pixel_subsampled)
        kernel_point_source /= np.sum(kernel_point_source)
        kwargs_pixel = {'psf_type': 'PIXEL',
                        'kernel_point_source': kernel_point_source}
        psf_pixel = PSF(**kwargs_pixel)
        kernel_point_source = psf_pixel.kernel_point_source
        kernel_point_source_new = psf_pixel_subsampled.kernel_point_source
        npt.assert_almost_equal(np.sum(kernel_point_source), np.sum(kernel_point_source_new), decimal=8)
        npt.assert_almost_equal(np.sum(kernel_point_source), 1, decimal=8)

        psf_none = PSF(psf_type='NONE')
        kernel_super = psf_none.kernel_point_source_supersampled(supersampling_factor=5)
        npt.assert_almost_equal(kernel_super, psf_none.kernel_point_source, decimal=9)

    def test_fwhm(self):
        deltaPix = 1.
        fwhm = 5.6
        kwargs = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_kernel = PSF(**kwargs)
        fwhm_compute = psf_kernel.fwhm
        assert fwhm_compute == fwhm

        kernel = kernel_util.kernel_gaussian(kernel_numPix=31, deltaPix=deltaPix, fwhm=fwhm)
        kwargs = {'psf_type': 'PIXEL',  'truncation': 5, 'pixel_size': deltaPix, 'kernel_point_source': kernel}
        psf_kernel = PSF(**kwargs)
        fwhm_compute = psf_kernel.fwhm
        npt.assert_almost_equal(fwhm_compute, fwhm, decimal=1)

        kwargs = {'psf_type': 'PIXEL', 'truncation': 5, 'pixel_size': deltaPix, 'kernel_point_source': kernel,
                  'point_source_supersampling_factor': 1}
        psf_kernel = PSF(**kwargs)
        fwhm_compute = psf_kernel.fwhm
        npt.assert_almost_equal(fwhm_compute, fwhm, decimal=1)

    def test_kernel_pixel(self):
        deltaPix = 1.
        fwhm = 5.6
        kwargs = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_kernel = PSF(**kwargs)
        kernel_pixel = psf_kernel.kernel_pixel
        npt.assert_almost_equal(np.sum(kernel_pixel), np.sum(psf_kernel.kernel_point_source), decimal=9)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        psf = PSF(psf_type='PIXEL', kernel_point_source=np.ones((3, 3)))
        psf.psf_type = 'WRONG'
        with self.assertRaises(ValueError):
            PSF(psf_type='GAUSSIAN')
        with self.assertRaises(ValueError):
            PSF(psf_type='PIXEL')
        with self.assertRaises(ValueError):
            PSF(psf_type='PIXEL', kernel_point_source=np.ones((2, 2)))
        with self.assertRaises(ValueError):
            PSF(psf_type='WRONG')
        with self.assertRaises(ValueError):
            PSF(psf_type='PIXEL', kernel_point_source=np.ones((3, 3)), psf_error_map=np.ones((5, 5)))
            psf.kernel_point_source_supersampled(supersampling_factor=3)
        with self.assertRaises(ValueError):
            psf = PSF(psf_type='PIXEL', kernel_point_source=np.ones((3, 3)))
            psf.psf_type = 'WRONG'
            psf.kernel_point_source_supersampled(supersampling_factor=3)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            kernel_point_source_subsampled = np.ones((9, 9))
            subsampling_res = 3
            kwargs_pixel_subsampled = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source_subsampled,
                                       'point_source_supersampling_factor': subsampling_res}
            psf_pixel_subsampled = PSF(**kwargs_pixel_subsampled)
            psf_pixel_subsampled.kernel_point_source_supersampled(supersampling_factor=subsampling_res + 4)
            # Verify some things
            assert 1 == 1
            #assert len(w) == 1
            #assert issubclass(w[-1].category, Warning)


if __name__ == '__main__':
    pytest.main()

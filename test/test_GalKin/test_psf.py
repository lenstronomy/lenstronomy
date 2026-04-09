from lenstronomy.GalKin.psf import (
    PSF,
    _fwhm_from_radial_profile,
    _radial_profile_from_kernel,
)
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.Util import kernel_util
import numpy as np
import numpy.testing as npt
import unittest


class TestPSF(object):
    def setup_method(self):
        self.fwhm = 0.7
        self.sigma = util.fwhm2sigma(self.fwhm)
        self.delta_pix = 0.3
        self.num_pix = 21
        self.data = np.arange(100).reshape(10, 10).astype(float)
        self.gaussian_kernel = kernel_util.kernel_gaussian(
            self.num_pix, self.delta_pix, self.fwhm
        )

    def test_displace_psf(self):
        psf = PSF(psf_type="GAUSSIAN", fwhm=1)
        np.random.seed(41)
        x, y = psf.displace_psf(0, 0)
        assert x != 0
        assert y != 0

        psf = PSF(psf_type="MOFFAT", fwhm=1, moffat_beta=2.6)
        np.random.seed(41)
        x, y = psf.displace_psf(0, 0)
        assert x != 0
        assert y != 0

    def test_kernel(self):
        psf = PSF(psf_type="GAUSSIAN", fwhm=1)
        kernel = psf.convolution_kernel(delta_pix=0.3, num_pix=21)
        npt.assert_almost_equal(np.sum(kernel), 1, decimal=5)
        x, y = util.make_grid(numPix=21, deltapix=0.3)
        kernel_direct = psf.convolution_kernel_grid(x, y).reshape(21, 21)
        npt.assert_equal(kernel, kernel_direct)

        psf = PSF(psf_type="MOFFAT", fwhm=1, moffat_beta=2.6)
        kernel = psf.convolution_kernel(delta_pix=0.3, num_pix=21)
        npt.assert_almost_equal(np.sum(kernel), 1, decimal=5)
        x, y = util.make_grid(numPix=21, deltapix=0.3)
        kernel_direct = psf.convolution_kernel_grid(x, y).reshape(21, 21)
        npt.assert_equal(kernel, kernel_direct)

        psf = PSF(
            psf_type="MULTI_GAUSSIAN",
            amplitudes=np.array([0.6, 0.4]),
            sigmas=np.array([0.3, 0.8]),
        )
        kernel = psf.convolution_kernel(delta_pix=0.3, num_pix=21)
        npt.assert_almost_equal(np.sum(kernel), 1, decimal=5)
        x, y = util.make_grid(numPix=21, deltapix=0.3)
        kernel_direct = psf.convolution_kernel_grid(x, y).reshape(21, 21)
        npt.assert_almost_equal(np.sum(kernel_direct), 1, decimal=5)
        npt.assert_allclose(kernel, kernel_direct, rtol=1e-5)

        psf = PSF(
            psf_type="PIXEL",
            kernel=self.gaussian_kernel,
            delta_pix=0.3,
            supersampling_factor=1,
        )
        kernel = psf.convolution_kernel(delta_pix=0.3, num_pix=21)
        npt.assert_almost_equal(np.sum(kernel), 1, decimal=5)
        npt.assert_almost_equal(kernel, self.gaussian_kernel, decimal=5)
        x, y = util.make_grid(numPix=21, deltapix=0.3)
        kernel_direct = psf.convolution_kernel_grid(
            x.reshape(21, 21), y.reshape(21, 21)
        )
        npt.assert_equal(kernel, kernel_direct)
        # approximated pixelated psf FWHM
        npt.assert_allclose(psf.psf_fwhm, self.fwhm, rtol=1e-1)

    def test_moffat_multi_gaussian_approx(self):
        moffat = PSF(psf_type="MOFFAT", fwhm=1, moffat_beta=2)
        moffat_kernel = moffat.convolution_kernel(delta_pix=0.1, num_pix=31)

        multi_gauss = PSF(
            psf_type="MULTI_GAUSSIAN",
            amplitudes=moffat.psf_multi_gauss_amplitudes,
            sigmas=moffat.psf_multi_gauss_sigmas,
        )
        multi_gauss_kernel = multi_gauss.convolution_kernel(delta_pix=0.1, num_pix=31)
        npt.assert_allclose(multi_gauss_kernel, moffat_kernel, rtol=1e-2, atol=1e-4)

    def test_psf_props(self):
        psf = PSF(psf_type="GAUSSIAN", fwhm=1)
        assert psf.psf_fwhm == 1
        assert psf.psf_multi_gauss_amplitudes == 1
        npt.assert_allclose(psf.psf_multi_gauss_sigmas, util.fwhm2sigma(1), rtol=1e-3)

        psf = PSF(psf_type="MOFFAT", fwhm=1, moffat_beta=0.3)
        assert psf.psf_fwhm == 1
        npt.assert_almost_equal(
            psf.psf_multi_gauss_amplitudes,
            [0.00000e00, 4.08999e-04, 1.61160e-02, 2.29624e-02, 9.60513e-01],
            decimal=5,
        )
        npt.assert_almost_equal(
            psf.psf_multi_gauss_sigmas,
            [0.02817, 0.07937, 0.22361, 0.62996, 1.77477],
            decimal=5,
        )

        psf = PSF(
            psf_type="MULTI_GAUSSIAN",
            amplitudes=np.array([0.6, 0.4]),
            sigmas=np.array([0.3, 0.8]),
        )
        npt.assert_allclose(psf.psf_fwhm, 0.745, rtol=1e-1)
        npt.assert_almost_equal(psf.psf_multi_gauss_amplitudes, [0.6, 0.4], decimal=5)
        npt.assert_almost_equal(psf.psf_multi_gauss_sigmas, [0.3, 0.8], decimal=5)

        psf = PSF(
            psf_type="PIXEL",
            kernel=self.gaussian_kernel,
            delta_pix=0.3,
            supersampling_factor=2,
        )
        npt.assert_allclose(psf.psf_fwhm, self.fwhm, rtol=1e-1)
        assert psf.psf_multi_gauss_amplitudes is None
        assert psf.psf_multi_gauss_sigmas is None
        assert psf._psf.supersampling_factor == 2

    def test_fwhm_from_radial_profile(self):
        r = np.linspace(0, 3, 100)
        p = Gaussian().function(r, y=0, amp=1, sigma=self.sigma, center_x=0, center_y=0)
        fwhm = _fwhm_from_radial_profile(r, p)
        npt.assert_allclose(fwhm, self.fwhm, rtol=1e-3)

    def test_radial_profile_from_kernel(self):
        r, p = _radial_profile_from_kernel(self.gaussian_kernel, self.delta_pix)
        npt.assert_allclose(
            p,
            Gaussian().function(r, 0, 1, self.sigma) * self.delta_pix**2,
            rtol=0.3,
            atol=1e-3,
        )
        fwhm = _fwhm_from_radial_profile(r, p)
        npt.assert_allclose(fwhm, self.fwhm, rtol=1e-1)


class TestRaise(unittest.TestCase):
    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            psf = PSF(psf_type="BRRR", fwhm=1, moffat_beta=2.6)
            # psf.displace_psf(0, 0)

    def test_invalid_pixel_kernel_not_square(self):
        with self.assertRaises(ValueError, msg="kernel must be square"):
            kernel = np.ones((5, 4))
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )

    def test_invalid_pixel_kernel_not_odd(self):
        with self.assertRaises(ValueError, msg="kernel must be odd-sized"):
            kernel = np.ones((4, 4))
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )

    def test_pixel_kernel_not_match_shape(self):
        with self.assertRaises(ValueError, msg="PSF grid does not match kernel shape"):
            kernel = np.zeros((5, 5))
            kernel[2, 2] = 1.0
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )
            psf.convolution_kernel(delta_pix=0.3, num_pix=7)

    def test_pixel_kernel_not_match_scale(self):
        with self.assertRaises(
            ValueError, msg="PSF delta_pix does not match kernel pixel scale"
        ):
            kernel = np.zeros((5, 5))
            kernel[2, 2] = 1.0
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )
            psf.convolution_kernel(delta_pix=0.1, num_pix=5)

    def test_pixel_kernel_not_match_shape_grid(self):
        with self.assertRaises(ValueError, msg="PSF grid does not match kernel shape"):
            kernel = np.zeros((5, 5))
            kernel[2, 2] = 1.0
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )
            x, y = util.make_grid(numPix=7, deltapix=0.3)
            x, y = x.reshape(7, 7), y.reshape(7, 7)
            psf.convolution_kernel_grid(x, y)

    def test_pixel_displacement_not_implemented(self):
        with self.assertRaises(NotImplementedError, msg="displace_psf not implemented"):
            kernel = np.zeros((5, 5))
            kernel[2, 2] = 1.0
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )
            psf.displace_psf(0, 0)

    def test_multi_gauss_displacement_not_implemented(self):
        with self.assertRaises(NotImplementedError, msg="displace_psf not implemented"):
            kernel = np.zeros((5, 5))
            kernel[2, 2] = 1.0
            psf = PSF(
                psf_type="PIXEL", kernel=kernel, delta_pix=0.3, supersampling_factor=1
            )
            psf.displace_psf(0, 0)


if __name__ == "__main__":
    unittest.main()

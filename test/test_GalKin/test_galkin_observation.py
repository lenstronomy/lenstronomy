from lenstronomy.GalKin.observation import GalkinObservation
from lenstronomy.Util.kernel_util import kernel_gaussian
import numpy as np
from numpy import testing as npt
from scipy.signal import convolve2d
import pytest


class TestGalkinObservation(object):

    def setup_method(self):
        self.kwargs_aperture_slit = {
            "aperture_type": "slit",
            "length": 3.0,
            "width": 0.5,
            "center_ra": 0,
            "center_dec": 0,
            "angle": np.deg2rad(30),
        }
        x = y = np.linspace(-5, 5, 20)
        self.grid_delta_pix = x[1] - x[0]
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.kwargs_aperture_grid = {
            "aperture_type": "IFU_grid",
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
        }
        self.kwargs_psf_gauss = {
            "psf_type": "GAUSSIAN",
            "fwhm": 0.5,
        }
        self.pix_kernel = np.zeros((5, 5))
        self.pix_kernel[2, 2] = 1.0
        self.kwargs_psf_pixel = {
            "psf_type": "PIXEL",
            "fwhm": 0.1,
            "kernel": self.pix_kernel,
            "supersampling_factor": 3,
            "delta_pix": (x[1] - x[0]) / 3,
        }

    def test_delta_pix(self):
        obs_slit = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="galkin",
        )
        assert obs_slit.delta_pix == 0.1

        obs_grid = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_grid,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="galkin",
        )
        assert obs_grid.delta_pix == self.grid_delta_pix

    def test_padding(self):
        obs_galkin = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="galkin",
        )
        npt.assert_almost_equal(obs_galkin._aperture.padding_arcsec, 0.634, decimal=2)

        obs_jampy = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="jampy",
        )
        assert obs_jampy._aperture.padding_arcsec == 0

        obs_pix = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_pixel,
            backend="jampy",
        )
        npt.assert_almost_equal(obs_pix._aperture.padding_arcsec, 0.127, decimal=2)

    def test_default_supersampling(self):
        obs_gauss = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="galkin",
        )
        assert obs_gauss._default_supersampling_factor == 1

        obs_pix = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_pixel,
            backend="galkin",
        )
        assert obs_pix._default_supersampling_factor == 3

    def test_convolve(self):
        data = np.zeros((9, 9))
        data[3, 3] = 1

        obs_gauss = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_slit,
            kwargs_psf=self.kwargs_psf_gauss,
            backend="galkin",
        )

        data_gauss = convolve2d(data, kernel_gaussian(31, 0.1, 0.5), mode="same")
        obs_conv = obs_gauss.convolve(data)
        npt.assert_allclose(obs_conv, data_gauss, rtol=1e-3)

        obs_pix = GalkinObservation(
            kwargs_aperture=self.kwargs_aperture_grid,
            kwargs_psf=self.kwargs_psf_pixel,
            backend="jampy",
        )
        data_pix = convolve2d(data, self.pix_kernel, mode="same")
        obs_conv = obs_pix.convolve(data, supersampling_factor=3)
        npt.assert_allclose(obs_conv, data_pix, rtol=1e-3)


if __name__ == "__main__":
    pytest.main()

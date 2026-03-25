from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF

import numpy as np
import numpy.testing as npt
import pytest
import unittest


class TestPointSourceRendering(object):
    def setup_method(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": 0,
            "dec_at_xy_0": 0,
            "transform_pix2angle": Mpix2coord,
            "nx": 10,
            "ny": 10,
        }
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {
            "kernel_point_source": kernel,
            "psf_type": "PIXEL",
            "psf_variance_map": np.ones_like(kernel) * kernel**2,
        }
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(
            pixel_grid, supersampling_factor=1, psf=psf_class
        )

    def test_psf_variance_map(self):
        ra_pos, dec_pos = [5], [5]
        data = np.zeros((10, 10))
        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        npt.assert_almost_equal(np.sum(image), 0, decimal=10)

        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=True
        )
        npt.assert_almost_equal(np.sum(image), 1, decimal=10)

        ra_pos, dec_pos = [50], [50]
        data = np.zeros((10, 10))
        image = self._ps_rendering.psf_variance_map(
            ra_pos, dec_pos, amp=1, data=data, fix_psf_variance_map=False
        )
        npt.assert_almost_equal(np.sum(image), 0, decimal=10)

    def test_point_source_rendering(self):
        amp = [1, 1]
        ra_pos, dec_pos = [0, 1], [1, 0]
        model = self._ps_rendering.point_source_rendering(ra_pos, dec_pos, amp)
        npt.assert_almost_equal(np.sum(model), 2, decimal=8)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": 0,
            "dec_at_xy_0": 0,
            "transform_pix2angle": Mpix2coord,
            "nx": 10,
            "ny": 10,
        }
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {
            "kernel_point_source": kernel,
            "psf_type": "PIXEL",
            "psf_variance_map": np.ones_like(kernel),
        }
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(
            pixel_grid, supersampling_factor=1, psf=psf_class
        )
        with self.assertRaises(ValueError):
            self._ps_rendering.point_source_rendering(
                ra_pos=[1, 1], dec_pos=[0, 1], amp=[1]
            )


class TestPointSourceRendering_for_interfermetry(unittest.TestCase):
    def test_point_source_rendering_unconvolved_for_interferometry(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": -2.5,
            "dec_at_xy_0": -2.5,
            "transform_pix2angle": Mpix2coord,
            "nx": 5,
            "ny": 5,
        }
        pixel_grid = PixelGrid(**kwargs_grid)

        # test the unconvolved rendering using a random PSF
        np.random.seed(42)
        kernel_random = np.random.uniform(0.2, 1, (5, 5))
        kernel_random /= kernel_random.sum()
        kwargs_psf_random = {
            "kernel_point_source": kernel_random,
            "psf_type": "PIXEL",
        }
        psf_class_random = PSF(**kwargs_psf_random)

        kernel_uncon = np.zeros((3, 3))
        kernel_uncon[1, 1] = 1
        kwargs_psf_uncon = {
            "kernel_point_source": kernel_uncon,
            "psf_type": "PIXEL",
        }
        psf_class_uncon = PSF(**kwargs_psf_uncon)

        ps_rendering_random_psf = PointSourceRendering(
            pixel_grid, supersampling_factor=None, psf=psf_class_random
        )
        ps_rendering_uncon_psf = PointSourceRendering(
            pixel_grid, supersampling_factor=None, psf=psf_class_uncon
        )

        amp = [2, 1]
        ra_pos, dec_pos = [-0.75, 0.17], [-0.03, 0.02]
        model_random_psf = ps_rendering_random_psf.point_source_rendering_unconvolved_for_interferometry(
            ra_pos, dec_pos, amp
        )
        model_uncon = ps_rendering_uncon_psf.point_source_rendering_unconvolved_for_interferometry(
            ra_pos, dec_pos, amp
        )
        npt.assert_almost_equal(model_random_psf, model_uncon, decimal=8)
        npt.assert_almost_equal(np.sum(model_uncon), 3.0, decimal=8)

        # test raise error for incorrect amp input size
        with self.assertRaises(ValueError):
            ps_rendering_uncon_psf.point_source_rendering_unconvolved_for_interferometry(
                ra_pos=[1, 1], dec_pos=[0, 1], amp=[1]
            )

    def test_raise_error_for_interferometric_supersampling(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {
            "ra_at_xy_0": -2.5,
            "dec_at_xy_0": -2.5,
            "transform_pix2angle": Mpix2coord,
            "nx": 5,
            "ny": 5,
        }
        pixel_grid = PixelGrid(**kwargs_grid)

        kernel_uncon = np.zeros((3, 3))
        kernel_uncon[1, 1] = 1
        kwargs_psf = {
            "kernel_point_source": kernel_uncon,
            "psf_type": "PIXEL",
        }
        psf_class = PSF(**kwargs_psf)

        kwargs_psf_supersampling = {
            "kernel_point_source": kernel_uncon,
            "psf_type": "PIXEL",
            "point_source_supersampling_factor": 2,
        }
        psf_class_supersampling = PSF(**kwargs_psf_supersampling)

        # PS rendering class for two types of super sampling
        ps_rendering_supersampling = PointSourceRendering(
            pixel_grid, supersampling_factor=2, psf=psf_class
        )
        ps_rendering_supersampling_psf = PointSourceRendering(
            pixel_grid, supersampling_factor=None, psf=psf_class_supersampling
        )

        with self.assertRaises(ValueError):
            ps_rendering_supersampling.point_source_rendering_unconvolved_for_interferometry(
                ra_pos=[1, 1], dec_pos=[0, 1], amp=[1, 1]
            )

        with self.assertRaises(ValueError):
            ps_rendering_supersampling_psf.point_source_rendering_unconvolved_for_interferometry(
                ra_pos=[1, 1], dec_pos=[0, 1], amp=[1, 1]
            )


if __name__ == "__main__":
    pytest.main()

from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF

import numpy as np
import numpy.testing as npt
import pytest


class TestPointSourceRendering(object):

    def setup(self):
        Mpix2coord = np.array([[1, 0], [0, 1]])
        kwargs_grid = {'ra_at_xy_0': 0, 'dec_at_xy_0': 0,
                       'transform_pix2angle': Mpix2coord, 'nx': 10, 'ny': 10}
        pixel_grid = PixelGrid(**kwargs_grid)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {'kernel_point_source': kernel, 'psf_type': 'PIXEL', 'psf_error_map': np.ones_like(kernel)}
        psf_class = PSF(**kwargs_psf)

        self._ps_rendering = PointSourceRendering(pixel_grid, supersampling_factor=1, psf=psf_class)

    def test_psf_error_map(self):
        ra_pos, dec_pos = [5], [5]
        data = np.zeros((10, 10))
        image = self._ps_rendering.psf_error_map(ra_pos, dec_pos, amp=1, data=data, fix_psf_error_map=False)
        npt.assert_almost_equal(np.sum(image), 0, decimal=10)

        image = self._ps_rendering.psf_error_map(ra_pos, dec_pos, amp=1, data=data, fix_psf_error_map=True)
        npt.assert_almost_equal(np.sum(image), 1, decimal=10)

        ra_pos, dec_pos = [50], [50]
        data = np.zeros((10, 10))
        image = self._ps_rendering.psf_error_map(ra_pos, dec_pos, amp=1, data=data, fix_psf_error_map=False)
        npt.assert_almost_equal(np.sum(image), 0, decimal=10)

if __name__ == '__main__':
    pytest.main()

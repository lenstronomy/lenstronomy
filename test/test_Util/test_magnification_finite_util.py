import numpy.testing as npt
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_size, auto_raytracing_grid_resolution
import pytest

class TestMagnificationFiniteUtil(object):

    def test_grid_resolution(self):

        source_size = 1.
        res_scale = 0.03
        ref = 15.
        power = 1.2
        resolution = auto_raytracing_grid_resolution(source_size, res_scale, ref, power)

        npt.assert_almost_equal(resolution, res_scale * (source_size/ref) ** power)

    def test_grid_size(self):

        source_size = 20.
        size = auto_raytracing_grid_size(source_size, grid_size_scale=0.04, power=0.9)

        npt.assert_almost_equal(size, 0.04 * source_size ** 0.9)

if __name__ == '__main__':
    pytest.main()

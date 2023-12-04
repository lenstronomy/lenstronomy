from lenstronomy.Util.magnification_finite_util import (
    auto_raytracing_grid_size,
    auto_raytracing_grid_resolution,
    setup_mag_finite,
)
from lenstronomy.LensModel.lens_model import LensModel
import numpy.testing as npt
from lenstronomy.LightModel.light_model import LightModel
import pytest


class TestMagnificationFiniteUtil(object):
    def test_grid_resolution(self):
        source_size = 1.0
        res_scale = 0.03
        ref = 15.0
        power = 1.2
        resolution = auto_raytracing_grid_resolution(source_size, res_scale, ref, power)

        npt.assert_almost_equal(resolution, res_scale * (source_size / ref) ** power)

    def test_grid_size(self):
        source_size = 20.0
        size = auto_raytracing_grid_size(source_size, grid_size_scale=0.04, power=0.9)

        npt.assert_almost_equal(size, 0.04 * source_size**0.9)

    def test_setup(self):
        grid_resolution = 0.001
        grid_radius_arcsec = 0.05
        source_light_model = ["GAUSSIAN"]
        source_model = LightModel(source_light_model)
        kwargs_source = [{'amp': 1, 'sigma': 0.0408 ,'center_x': 0, 'center_y':0}]


        npt.assert_equal(True, grid_resolution is not None)
        npt.assert_equal(True, grid_radius_arcsec is not None)

        npt.assert_equal(True, grid_resolution == 0.001)
        npt.assert_equal(True, grid_radius_arcsec == 0.05)

        (
            gridx,
            gridy,
            source_model,
            kwargs_source,
            grid_resolution,
            grid_radius_arcsec,
        ) = setup_mag_finite(
            grid_radius_arcsec,
            grid_resolution,
            source_model,
            kwargs_source
        )



if __name__ == "__main__":
    pytest.main()

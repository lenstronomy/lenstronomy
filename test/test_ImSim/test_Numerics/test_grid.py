__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
from lenstronomy.Util import util
from lenstronomy.ImSim.Numerics.grid import AdaptiveGrid
from lenstronomy.ImSim.Numerics.grid import RegularGrid
from lenstronomy.LightModel.light_model import LightModel

import pytest


class TestAdaptiveGrid(object):
    def setup_method(self):
        deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        nx, ny = 11, 11
        self._supersampling_factor = 4
        supersampling_indexes = np.zeros((nx, ny))
        supersampling_indexes = np.array(supersampling_indexes, dtype=bool)
        supersampling_indexes[5, 5] = True
        self._supersampling_indexes = supersampling_indexes
        self.nx, self.ny = nx, ny
        self._adaptive_grid = AdaptiveGrid(
            nx,
            ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_indexes,
            self._supersampling_factor,
        )

    def test_coordinates_evaluate(self):
        x_grid, y_grid = self._adaptive_grid.coordinates_evaluate
        print(np.shape(x_grid), "test shape")
        assert len(x_grid) == self._supersampling_factor**2 + self.nx * self.ny - 1

    def test_subpixel_coordinates(self):
        subpixel_x, subpixel_y = self._adaptive_grid._high_res_coordinates
        assert len(subpixel_x) == 4**2
        assert subpixel_x[0] == -0.375
        assert subpixel_y[0] == -0.375
        assert subpixel_y[3] == -0.375
        assert subpixel_x[3] == 0.375

    def test_average_subgrid(self):
        subpixel_x, subpixel_y = self._adaptive_grid._high_res_coordinates
        model = LightModel(light_model_list=["GAUSSIAN"])
        kwargs_light = [{"center_x": 0, "center_y": 0, "sigma": 1, "amp": 1}]
        subgrid_values = model.surface_brightness(subpixel_x, subpixel_y, kwargs_light)
        supersampled_values = self._adaptive_grid._average_subgrid(subgrid_values)
        assert len(supersampled_values) == 1

    def test_merge_low_high_res(self):
        subpixel_x, subpixel_y = self._adaptive_grid._high_res_coordinates
        x, y = self._adaptive_grid._x_low_res, self._adaptive_grid._x_low_res
        model = LightModel(light_model_list=["GAUSSIAN"])
        kwargs_light = [{"center_x": 0, "center_y": 0, "sigma": 1, "amp": 1}]
        subgrid_values = model.surface_brightness(subpixel_x, subpixel_y, kwargs_light)
        image1d = model.surface_brightness(x, y, kwargs_light)

        image_added = self._adaptive_grid._merge_low_high_res(image1d, subgrid_values)
        added_array = util.image2array(image_added)
        supersampled_values = self._adaptive_grid._average_subgrid(subgrid_values)
        assert (
            added_array[util.image2array(self._supersampling_indexes)]
            == supersampled_values
        )

        image_high_res = self._adaptive_grid._high_res_image(subgrid_values)
        assert len(image_high_res) == self.nx * self._supersampling_factor

    def test_flux_array2image_low_high(self):
        x, y = self._adaptive_grid.coordinates_evaluate
        model = LightModel(light_model_list=["GAUSSIAN"])
        kwargs_light = [{"center_x": 0, "center_y": 0, "sigma": 1, "amp": 1}]
        flux_values = model.surface_brightness(x, y, kwargs_light)
        image_low_res, image_high_res = self._adaptive_grid.flux_array2image_low_high(
            flux_values
        )
        assert len(image_high_res) == self.nx * self._supersampling_factor


class TestRegularGrid(object):
    def setup_method(self):
        self._deltaPix = 1.0
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self._deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        nx, ny = 11, 11
        self._supersampling_factor = 4
        self.nx, self.ny = nx, ny
        self._regular_grid = RegularGrid(
            nx,
            ny,
            transform_pix2angle,
            ra_at_xy_0,
            dec_at_xy_0,
            supersampling_factor=self._supersampling_factor,
        )

    def test_grid_points_spacing(self):
        deltaPix = self._regular_grid.grid_points_spacing
        assert deltaPix == self._deltaPix / self._supersampling_factor

    def test_num_grid_points_axes(self):
        nx, ny = self._regular_grid.num_grid_points_axes
        assert nx == self.nx * self._supersampling_factor
        assert ny == self.ny * self._supersampling_factor

    def test_supersampling_factor(self):
        ssf = self._regular_grid.supersampling_factor
        assert ssf == self._supersampling_factor


if __name__ == "__main__":
    pytest.main()

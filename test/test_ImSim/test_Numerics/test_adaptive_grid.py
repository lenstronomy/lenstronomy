__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
from lenstronomy.Util import util
from lenstronomy.ImSim.Numerics.adaptive_grid import AdaptiveGrid
from lenstronomy.LightModel.light_model import LightModel

import pytest


class TestAdaptiveGrid(object):

    def setup(self):
        deltaPix = 1
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_at_xy_0, dec_at_xy_0 = -5, -5
        nx, ny = 11, 11
        self._supersampling_factor = 4
        supersampling_indexes = np.zeros((nx, ny))
        supersampling_indexes = np.array(supersampling_indexes, dtype=bool)
        supersampling_indexes[5, 5] = True
        self._supersampling_indexes = supersampling_indexes
        self.nx, self.ny = nx, ny
        self._adaptive_grid = AdaptiveGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_indexes, self._supersampling_factor)

    def test_subpixel_coordinates(self):
        subpixel_x, subpixel_y = self._adaptive_grid.subpixel_coordinates
        assert len(subpixel_x) == 4**2
        assert subpixel_x[0] == -0.375
        assert subpixel_y[0] == -0.375
        assert subpixel_y[3] == 0.375
        assert subpixel_x[3] == -0.375

    def test_average_subgrid(self):
        subpixel_x, subpixel_y = self._adaptive_grid.subpixel_coordinates
        model = LightModel(light_model_list=['GAUSSIAN'])
        kwargs_light = [{'center_x': 0, 'center_y': 0, 'sigma_x': 1, 'sigma_y': 1, 'amp': 1}]
        subgrid_values = model.surface_brightness(subpixel_x, subpixel_y, kwargs_light)
        supersampled_values = self._adaptive_grid.average_subgrid(subgrid_values)
        assert len(supersampled_values) == 1

        image = np.zeros((self.nx, self.ny))
        image1d = util.image2array(image)
        image_added = self._adaptive_grid.add_supersampled(image1d, supersampled_values)
        assert image_added[util.image2array(self._supersampling_indexes)] == supersampled_values

        image_high_res = self._adaptive_grid.high_res_image(subgrid_values)
        assert len(image_high_res) == self.nx * self._supersampling_factor


if __name__ == '__main__':
    pytest.main()

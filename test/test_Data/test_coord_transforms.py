import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.Data.coord_transforms import Coordinates
import lenstronomy.Util.util as util


class TestCoordinates(object):
    def setup(self):
        pass

    def test_init(self):
        deltaPix = 0.05
        Mpix2a = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_0 = 1.
        dec_0 = 1.
        coords = Coordinates(transform_pix2angle=Mpix2a, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0)
        ra, dec = coords.map_pix2coord(0, 0)
        assert ra == ra_0
        assert dec == dec_0
        x, y = coords.map_coord2pix(ra, dec)
        assert ra_0 == ra
        assert dec_0 == dec
        assert x == 0
        assert y == 0

    def test_map_coord2pix(self):
        deltaPix = 0.05
        Mpix2a = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_0 = 1.
        dec_0 = 1.
        coords = Coordinates(transform_pix2angle=Mpix2a, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0)
        x, y = coords.map_coord2pix(2, 1)
        assert x == 20
        assert y == 0

    def test_map_pix2coord(self):
        deltaPix = 0.05
        Mpix2a = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_0 = 1.
        dec_0 = 1.
        coords = Coordinates(transform_pix2angle=Mpix2a, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0)
        x, y = coords.map_pix2coord(1, 0)
        assert x == deltaPix + ra_0
        assert y == dec_0

    def test_pixel_size(self):
        deltaPix = -0.05
        Mpix2a = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_0 = 1.
        dec_0 = 1.
        coords = Coordinates(transform_pix2angle=Mpix2a, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0)
        deltaPix_out = coords.pixel_size
        assert deltaPix_out == -deltaPix

    def test_rescaled_grid(self):
        import lenstronomy.Util.util as util
        numPix = 10
        theta = 0.5
        deltaPix = 0.05
        subgrid_res = 3
        Mpix2a = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * deltaPix
        x_grid, y_grid = util.make_grid_transformed(numPix, Mpix2a)
        coords = Coordinates(Mpix2a, ra_at_xy_0=x_grid[0],
                                 dec_at_xy_0=y_grid[0])
        x_grid_high_res, y_grid_high_res = util.make_subgrid(x_grid, y_grid, subgrid_res=subgrid_res)
        coords_sub = Coordinates(Mpix2a/subgrid_res, ra_at_xy_0=x_grid_high_res[0],
                                    dec_at_xy_0=y_grid_high_res[0])

        x, y = coords_sub.map_coord2pix(x_grid[1], y_grid[1])
        npt.assert_almost_equal(x, 4, decimal=10)
        npt.assert_almost_equal(y, 1, decimal=10)
        x, y = coords_sub.map_coord2pix(x_grid[0], y_grid[0])
        npt.assert_almost_equal(x, 1, decimal=10)
        npt.assert_almost_equal(y, 1, decimal=10)

        ra, dec = coords_sub.map_pix2coord(1, 1)
        npt.assert_almost_equal(ra, x_grid[0], decimal=10)
        npt.assert_almost_equal(dec, y_grid[0], decimal=10)

        ra, dec = coords_sub.map_pix2coord(1 + 2*subgrid_res, 1)
        npt.assert_almost_equal(ra, x_grid[2], decimal=10)
        npt.assert_almost_equal(dec, y_grid[2], decimal=10)

        x_2d = util.array2image(x_grid)
        y_2d = util.array2image(y_grid)

        ra, dec = coords_sub.map_pix2coord(1 + 2*subgrid_res, 1 + 3*subgrid_res)
        npt.assert_almost_equal(ra, x_2d[3, 2], decimal=10)
        npt.assert_almost_equal(dec, y_2d[3, 2], decimal=10)

        ra, dec = coords.map_pix2coord(2, 3)
        npt.assert_almost_equal(ra, x_2d[3, 2], decimal=10)
        npt.assert_almost_equal(dec, y_2d[3, 2], decimal=10)

    def test_coordinate_grid(self):
        deltaPix = 0.05
        Mpix2a = np.array([[1, 0], [0, 1]]) * deltaPix
        ra_0 = 1.
        dec_0 = 1.
        coords = Coordinates(transform_pix2angle=Mpix2a, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0)
        ra_grid, dec_grid = coords.coordinate_grid(numPix=10)
        ra_grid_2d = util.array2image(ra_grid)
        dec_grid_2d = util.array2image(dec_grid)
        assert ra_grid[0] == ra_0
        assert dec_grid[0] == dec_0
        x_pos, y_pos = 1, 2
        ra, dec = coords.map_pix2coord(x_pos, y_pos)
        npt.assert_almost_equal(ra_grid_2d[int(y_pos), int(x_pos)], ra, decimal=8)
        npt.assert_almost_equal(dec_grid_2d[int(y_pos), int(x_pos)], dec, decimal=8)


if __name__ == '__main__':
    pytest.main()
import pytest
import numpy as np
import numpy.testing as npt
import numpy.linalg as linalg

from lenstronomy.Data.coord_transforms import Coordinates


class TestCoordinates(object):
    def setup_method(self):
        pass

    def test_init(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        ra, dec = coords.map_pix2coord(0, 0)
        assert ra == ra_0
        assert dec == dec_0
        x, y = coords.map_coord2pix(ra, dec)
        assert ra_0 == ra
        assert dec_0 == dec
        assert x == 0
        assert y == 0

    def test_map_coord2pix(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        x, y = coords.map_coord2pix(2, 1)
        assert x == 20
        assert y == 0

    def test_map_pix2coord(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        x, y = coords.map_pix2coord(1, 0)
        assert x == delta_pix + ra_0
        assert y == dec_0

    def test_pixel_size(self):
        delta_pix = -0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        delta_pix_out = coords.pixel_width
        assert delta_pix_out == -delta_pix

    def test_rescaled_grid(self):
        import lenstronomy.Util.util as util

        num_pix = 10
        theta = 0.5
        delta_pix = 0.05
        subgrid_res = 3
        transform_pix2angle = (
            np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            * delta_pix
        )
        x_grid, y_grid = util.make_grid_transformed(num_pix, transform_pix2angle)
        coords = Coordinates(
            transform_pix2angle, ra_at_xy_0=x_grid[0], dec_at_xy_0=y_grid[0]
        )
        x_grid_high_res, y_grid_high_res = util.make_subgrid(
            x_grid, y_grid, subgrid_res=subgrid_res
        )
        coords_sub = Coordinates(
            transform_pix2angle / subgrid_res,
            ra_at_xy_0=x_grid_high_res[0],
            dec_at_xy_0=y_grid_high_res[0],
        )

        x, y = coords_sub.map_coord2pix(x_grid[1], y_grid[1])
        npt.assert_almost_equal(x, 4, decimal=10)
        npt.assert_almost_equal(y, 1, decimal=10)
        x, y = coords_sub.map_coord2pix(x_grid[0], y_grid[0])
        npt.assert_almost_equal(x, 1, decimal=10)
        npt.assert_almost_equal(y, 1, decimal=10)

        ra, dec = coords_sub.map_pix2coord(1, 1)
        npt.assert_almost_equal(ra, x_grid[0], decimal=10)
        npt.assert_almost_equal(dec, y_grid[0], decimal=10)

        ra, dec = coords_sub.map_pix2coord(1 + 2 * subgrid_res, 1)
        npt.assert_almost_equal(ra, x_grid[2], decimal=10)
        npt.assert_almost_equal(dec, y_grid[2], decimal=10)

        x_2d = util.array2image(x_grid)
        y_2d = util.array2image(y_grid)

        ra, dec = coords_sub.map_pix2coord(1 + 2 * subgrid_res, 1 + 3 * subgrid_res)
        npt.assert_almost_equal(ra, x_2d[3, 2], decimal=10)
        npt.assert_almost_equal(dec, y_2d[3, 2], decimal=10)

        ra, dec = coords.map_pix2coord(2, 3)
        npt.assert_almost_equal(ra, x_2d[3, 2], decimal=10)
        npt.assert_almost_equal(dec, y_2d[3, 2], decimal=10)

    def test_coordinate_grid(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        ra_grid, dec_grid = coords.coordinate_grid(nx=10, ny=10)

        assert ra_grid[0, 0] == ra_0
        assert dec_grid[0, 0] == dec_0
        x_pos, y_pos = 1, 2
        ra, dec = coords.map_pix2coord(x_pos, y_pos)
        npt.assert_almost_equal(ra_grid[int(y_pos), int(x_pos)], ra, decimal=8)
        npt.assert_almost_equal(dec_grid[int(y_pos), int(x_pos)], dec, decimal=8)

    def test_xy_at_radec_0(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        x_at_radec_0, y_at_radec_0 = coords.xy_at_radec_0
        npt.assert_almost_equal(x_at_radec_0, -20, decimal=8)
        npt.assert_almost_equal(x_at_radec_0, -20, decimal=8)
        transform_angle2pix_ = coords.transform_angle2pix
        transform_angle2pix = linalg.inv(coords._transform_pix2angle)
        npt.assert_almost_equal(transform_angle2pix, transform_angle2pix_, decimal=8)

    def test_shift_coordinate_system(self):
        delta_pix = 0.05
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * delta_pix
        ra_0 = 1.0
        dec_0 = 1.0
        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        x0, y0 = coords.xy_at_radec_0
        coords.shift_coordinate_system(x_shift=delta_pix, y_shift=0, pixel_unit=False)
        x0_new, y0_new = coords.xy_at_radec_0
        assert x0_new == x0 - 1

        coords = Coordinates(
            transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_0, dec_at_xy_0=dec_0
        )
        x0, y0 = coords.xy_at_radec_0
        coords.shift_coordinate_system(x_shift=1, y_shift=0, pixel_unit=True)
        x0_new, y0_new = coords.xy_at_radec_0
        assert x0_new == x0 - 1


if __name__ == "__main__":
    pytest.main()

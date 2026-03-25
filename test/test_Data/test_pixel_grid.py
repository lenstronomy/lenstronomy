import numpy.testing as npt
import numpy as np

from lenstronomy.Data.pixel_grid import PixelGrid


class TestPixelGrid(object):

    def setup_method(self):
        self.nx, self.ny = 30, 20
        self.delta_pix = 0.1
        transform_pix2angle = np.array([[1, 0], [0, 1]]) * self.delta_pix
        ra_at_xy_0, dec_at_xy_0 = 0, 0
        self.pixel_grid = PixelGrid(
            nx=self.nx,
            ny=self.ny,
            transform_pix2angle=transform_pix2angle,
            ra_at_xy_0=ra_at_xy_0,
            dec_at_xy_0=dec_at_xy_0,
            antenna_primary_beam=None,
        )

    def test_num_pix(self):
        num_pix = self.pixel_grid.num_pixel
        npt.assert_almost_equal(num_pix, self.nx * self.ny)

    def test_num_pixel_axes(self):
        nx, ny = self.pixel_grid.num_pixel_axes
        assert nx == self.nx
        assert ny == self.ny

    def test_width(self):
        dx, dy = self.pixel_grid.width
        npt.assert_almost_equal(dx, self.nx * self.delta_pix)
        npt.assert_almost_equal(dy, self.ny * self.delta_pix)

    def test_pixel_coordinates(self):
        x_grid, y_grid = self.pixel_grid.pixel_coordinates
        print(np.shape(x_grid[0, :]))

        npt.assert_almost_equal(
            x_grid[0, :],
            np.linspace(start=0, stop=(self.nx - 1) * self.delta_pix, num=self.nx),
            decimal=4,
        )
        npt.assert_almost_equal(
            y_grid[:, 0],
            np.linspace(start=0, stop=(self.ny - 1) * self.delta_pix, num=self.ny),
            decimal=4,
        )

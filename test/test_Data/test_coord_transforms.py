import pytest
import numpy as np

from lenstronomy.Data.coord_transforms import Coordinates


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
        print(x, y)
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


if __name__ == '__main__':
    pytest.main()
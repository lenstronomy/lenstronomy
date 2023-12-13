from lenstronomy.GalKin.aperture import Aperture

import pytest
import unittest
import numpy as np


class TestAperture(object):
    def setup_method(self):
        pass

    def test_aperture_select(self):
        kwargs_slit = {
            "length": 2,
            "width": 0.5,
            "center_ra": 0,
            "center_dec": 0,
            "angle": 0,
        }
        slit = Aperture(aperture_type="slit", **kwargs_slit)
        bool, i = slit.aperture_select(ra=0.9, dec=0.2)
        assert bool is True
        bool, i = slit.aperture_select(ra=1.1, dec=0.2)
        assert bool is False
        assert slit.num_segments == 1

        kwargs_shell = {"r_in": 0.2, "r_out": 1.0, "center_ra": 0, "center_dec": 0}
        shell = Aperture(aperture_type="shell", **kwargs_shell)
        bool, i = shell.aperture_select(ra=0.9, dec=0)
        assert bool is True
        bool, i = shell.aperture_select(ra=1.1, dec=0)
        assert bool is False
        bool, i = shell.aperture_select(ra=0.1, dec=0)
        assert bool is False
        assert shell.num_segments == 1

        kwargs_boxhole = {
            "width_outer": 1,
            "width_inner": 0.5,
            "center_ra": 0,
            "center_dec": 0,
        }
        frame = Aperture(aperture_type="frame", **kwargs_boxhole)
        bool, i = frame.aperture_select(ra=0.4, dec=0)
        assert bool is True
        bool, i = frame.aperture_select(ra=0.2, dec=0)
        assert bool is False
        assert frame.num_segments == 1

        x_grid, y_grid = np.meshgrid(
            np.arange(-0.9, 0.95, 0.20),  # x-axis points to negative RA
            np.arange(-0.9, 0.95, 0.20),
        )
        kwargs_ifugrid = {"x_grid": x_grid, "y_grid": y_grid}
        frame = Aperture(aperture_type="IFU_grid", **kwargs_ifugrid)
        bool, i = frame.aperture_select(ra=0.95, dec=0.95)
        assert bool is True
        assert i == (9, 9)
        bool, i = frame.aperture_select(ra=5, dec=5)
        assert bool is False
        assert frame.num_segments == (10, 10)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            Aperture(aperture_type="wrong", kwargs_aperture={})


if __name__ == "__main__":
    pytest.main()

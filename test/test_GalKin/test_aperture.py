from lenstronomy.GalKin.aperture import Aperture

import pytest
import unittest


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


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            Aperture(aperture_type="wrong", kwargs_aperture={})


if __name__ == "__main__":
    pytest.main()

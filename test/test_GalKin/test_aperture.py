from lenstronomy.GalKin.aperture import Aperture

import pytest
import unittest


class TestAperture(object):

    def setup(self):
        pass

    def test_aperture_select(self):
        kwargs_slit = {'length': 2, 'width': 0.5, 'center_ra': 0, 'center_dec': 0, 'angle': 0}
        slit = Aperture(aperture_type='slit', **kwargs_slit)
        bool, i = slit.aperture_select(ra=0.9, dec=0.2)
        assert bool is True
        bool, i = slit.aperture_select(ra=1.1, dec=0.2)
        assert bool is False

        kwargs_shell = {'r_in': 0.2, 'r_out': 1., 'center_ra': 0, 'center_dec': 0}
        shell = Aperture(aperture_type='shell', **kwargs_shell)
        bool, i = shell.aperture_select(ra=0.9, dec=0)
        assert bool is True
        bool, i = shell.aperture_select(ra=1.1, dec=0)
        assert bool is False
        bool, i = shell.aperture_select(ra=0.1, dec=0)
        assert bool is False


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            Aperture(aperture_type='wrong', kwargs_aperture={})


if __name__ == '__main__':
    pytest.main()

from lenstronomy.GalKin.aperture import Aperture
import pytest


class TestAperture(object):

    def setup(self):
        pass

    def test_shell_select(self):
        aperture = Aperture()
        ra, dec = 1, 1
        r_in = 2
        r_out = 4
        bool_select = aperture.shell_select(ra, dec, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is False

        bool_select = aperture.shell_select(3, 0, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is True


if __name__ == '__main__':
    pytest.main()
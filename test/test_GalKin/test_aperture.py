from lenstronomy.GalKin.aperture import Aperture
from lenstronomy.GalKin import aperture
import pytest
import numpy as np


class TestAperture(object):

    def setup(self):
        pass

    def test_shell_select(self):
        #aperture = Aperture()
        ra, dec = 1, 1
        r_in = 2
        r_out = 4
        bool_select = aperture.shell_select(ra, dec, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is False

        bool_select = aperture.shell_select(3, 0, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is True

    def test_slit_select(self):
        bool_select = aperture.slit_select(ra=0.9, dec=0, length=2, width=0.5, center_ra=0, center_dec=0, angle=0)
        assert bool_select is True

        bool_select = aperture.slit_select(ra=0.9, dec=0, length=2, width=0.5, center_ra=0, center_dec=0, angle=np.pi/2)
        assert bool_select is False


if __name__ == '__main__':
    pytest.main()

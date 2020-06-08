from lenstronomy.GalKin import aperture_types
import pytest
import numpy as np


class TestApertureTypes(object):

    def setup(self):
        pass

    def test_shell_select(self):
        #aperture = Aperture()
        ra, dec = 1, 1
        r_in = 2
        r_out = 4
        bool_select = aperture_types.shell_select(ra, dec, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is False

        bool_select = aperture_types.shell_select(3, 0, r_in, r_out, center_ra=0, center_dec=0)
        assert bool_select is True

    def test_slit_select(self):
        bool_select = aperture_types.slit_select(ra=0.9, dec=0, length=2, width=0.5, center_ra=0, center_dec=0, angle=0)
        assert bool_select is True

        bool_select = aperture_types.slit_select(ra=0.9, dec=0, length=2, width=0.5, center_ra=0, center_dec=0, angle=np.pi/2)
        assert bool_select is False

    def test_ifu_shell_select(self):
        ra, dec = 1, 1
        r_bin = np.linspace(0, 10, 11)
        bool_select, i = aperture_types.shell_ifu_select(ra, dec, r_bin, center_ra=0, center_dec=0)
        assert bool_select is True
        assert i == 1

    def test_frame(self):
        center_ra, center_dec = 0, 0
        width_outer = 1.2
        width_inner = 0.6
        ra, dec = 0, 0
        bool_select = aperture_types.frame_select(ra, dec, width_inner=width_inner, width_outer=width_outer, center_ra=center_ra, center_dec=center_dec, angle=0)
        assert bool_select is False
        ra, dec = 0.5, 0
        bool_select = aperture_types.frame_select(ra, dec, width_inner=width_inner, width_outer=width_outer,
                                                  center_ra=center_ra, center_dec=center_dec, angle=0)
        assert bool_select is True
        ra, dec = 5, 5
        bool_select = aperture_types.frame_select(ra, dec, width_inner=width_inner, width_outer=width_outer,
                                                  center_ra=center_ra, center_dec=center_dec, angle=0)
        assert bool_select is False



if __name__ == '__main__':
    pytest.main()

__author__ = 'sibirrer'


import lenstronomy.Util.analysis_util as analysis_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import pytest


class TestCorrelation(object):

    def setup(self):
        pass

    def test_radial_profile(self):
        x_grid, y_grid = util.make_grid(numPix=20, deltapix=1)
        profile = Gaussian()
        light_grid = profile.function(x_grid, y_grid, amp=1., sigma_x=5, sigma_y=5)
        I_r, r = analysis_util.radial_profile(light_grid, x_grid, y_grid, center_x=0, center_y=0, n=None)
        assert I_r[0] == 0


if __name__ == '__main__':
    pytest.main()
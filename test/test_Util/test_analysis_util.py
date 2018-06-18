__author__ = 'sibirrer'


import lenstronomy.Util.analysis_util as analysis_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian, GaussianEllipse
import pytest
import numpy.testing as npt

class TestCorrelation(object):

    def setup(self):
        pass

    def test_radial_profile(self):
        x_grid, y_grid = util.make_grid(numPix=20, deltapix=1)
        profile = Gaussian()
        light_grid = profile.function(x_grid, y_grid, amp=1., sigma_x=5, sigma_y=5)
        I_r, r = analysis_util.radial_profile(light_grid, x_grid, y_grid, center_x=0, center_y=0, n=None)
        assert I_r[0] == 0

    def test_ellipticities(self):
        x_grid, y_grid = util.make_grid(numPix=200, deltapix=1)
        e1, e2 = 0., 0.1
        profile = GaussianEllipse()
        I_xy = profile.function(x_grid, y_grid, amp=1, sigma=10, e1=e1, e2=e2)
        e1_out, e2_out = analysis_util.ellipticities(I_xy, x_grid, y_grid)
        print(e1_out, e2_out)
        npt.assert_almost_equal(e1_out, e1, decimal=3)
        npt.assert_almost_equal(e2_out, e2, decimal=3)

        e1, e2 = 0.1, 0.
        profile = GaussianEllipse()
        I_xy = profile.function(x_grid, y_grid, amp=1, sigma=10, e1=e1, e2=e2)
        e1_out, e2_out = analysis_util.ellipticities(I_xy, x_grid, y_grid)
        print(e1_out, e2_out)
        npt.assert_almost_equal(e1_out, e1, decimal=3)
        npt.assert_almost_equal(e2_out, e2, decimal=3)



if __name__ == '__main__':
    pytest.main()
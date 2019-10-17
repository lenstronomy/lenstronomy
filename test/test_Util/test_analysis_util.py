__author__ = 'sibirrer'


import lenstronomy.Util.analysis_util as analysis_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian, GaussianEllipse
import pytest
import numpy.testing as npt
import numpy as np


class TestCorrelation(object):

    def setup(self):
        pass

    def test_radial_profile(self):
        x_grid, y_grid = util.make_grid(numPix=20, deltapix=1)
        profile = Gaussian()
        light_grid = profile.function(x_grid, y_grid, amp=1., sigma=5)
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

    def test_half_light_radius(self):
        x_grid, y_grid = util.make_grid(numPix=10, deltapix=1)
        lens_light = np.zeros_like(x_grid)
        r_half = analysis_util.half_light_radius(lens_light, x_grid, y_grid, center_x=0, center_y=0)
        assert r_half == -1

    def test_bic_model(self):
        bic=analysis_util.bic_model(0,np.e,1)
        assert bic == 1

    def test_azimuthalAverage(self):
        num_pix = 101
        x_grid, y_grid = util.make_grid(numPix=num_pix, deltapix=1)
        e1, e2 = 0., 0.
        profile = GaussianEllipse()
        kwargs_profile = {'amp': 1, 'sigma': 50, 'e1': e1, 'e2': e2}
        I_xy = profile.function(x_grid, y_grid, **kwargs_profile)
        I_xy = util.array2image(I_xy)
        I_r = analysis_util.azimuthalAverage(I_xy, center=None)
        r = np.linspace(start=0.5, stop=len(I_r) + 1 - 0.5, num=len(I_r))
        I_r_true = profile.function(0, r, **kwargs_profile)
        npt.assert_almost_equal(I_r / I_r_true, 1, decimal=2)

        r = np.sqrt(x_grid**2 + y_grid**2)
        r_max = np.max(r)
        I_xy = np.sin(r/r_max * (2*np.pi))
        I_xy = util.array2image(I_xy)
        I_r = analysis_util.azimuthalAverage(I_xy, center=None)
        r = np.linspace(start=0.5, stop=len(I_r) + 1 - 0.5, num=len(I_r))
        I_r_true = np.sin(r/r_max * (2*np.pi))
        #import matplotlib.pyplot as plt
        #plt.plot(r, I_r, label='computed')
        #plt.plot(r, I_r_true, label='true')
        #plt.legend()
        #plt.show()

        npt.assert_almost_equal(I_r[10:], I_r_true[10:], decimal=1)


if __name__ == '__main__':
    pytest.main()

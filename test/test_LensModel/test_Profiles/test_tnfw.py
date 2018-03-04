__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.nfw import NFW

import numpy as np
import numpy.testing as npt
import pytest

class TestTNFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()
        self.tnfw = TNFW()

    def test_derivatives(self):
        Rs = 1.
        theta_Rs = 0.5
        x_array = np.linspace(0, 10, 10)
        y_array = np.zeros_like(x_array)
        f_x, f_y = self.nfw.derivatives(x_array, y_array, Rs, theta_Rs)
        f_x_t, f_y_t = self.tnfw.derivatives(x_array, y_array, Rs, theta_Rs, r_trunc=1000.)
        #print(f_x/truth_alpha)
        print(f_x, f_x_t)
        for i in range(len(x_array)):
            npt.assert_almost_equal(f_x[i], f_x_t[i], decimal=3)
            npt.assert_almost_equal(f_y[i], f_y_t[i], decimal=3)

        f_x_t, f_y_t = self.tnfw.derivatives(1. , 1., Rs, theta_Rs, r_trunc=1.)
        npt.assert_almost_equal(f_x_t, 0.17145581715955596, decimal=8)


if __name__ == '__main__':
    pytest.main()

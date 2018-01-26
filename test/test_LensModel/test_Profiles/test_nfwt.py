__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfwT import NFWt

import numpy as np
import numpy.testing as npt
import pytest

class TestNFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()
        self.nfwt = NFWt()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_ = self.nfw.function(x, y, Rs, theta_Rs)
        t = 10000
        f_t = self.nfwt.function(x, y, Rs, theta_Rs, t)
        #npt.assert_almost_equal(f[0], f_t[0], decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_x, f_y = self.nfw.derivatives(x, y, Rs, theta_Rs)
        t = 10000
        f_xt, f_yt = self.nfwt.derivatives(x, y, Rs, theta_Rs, t)
        #npt.assert_almost_equal(f_xt[0], f_x[0], decimal=5)
        #npt.assert_almost_equal(f_yt[0], f_y[0], decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        t = 10000
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_xx, f_yy,f_xy = self.nfw.hessian(x, y, Rs, theta_Rs)
        f_xxt, f_yyt, f_xyt = self.nfwt.hessian(x, y, Rs, theta_Rs, t)
        #npt.assert_almost_equal(f_xx[0], f_xxt[0], decimal=5)
        #npt.assert_almost_equal(f_yy[0], f_yyt[0], decimal=5)
        #npt.assert_almost_equal(f_xy[0], f_xyt[0], decimal=5)


if __name__ == '__main__':
    pytest.main()
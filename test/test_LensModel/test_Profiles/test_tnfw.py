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


class TestTNFW_new(object):

    def setup(self):
        from lenstronomy.LensModel.Profiles.tnfw_new import TNFW
        self.nfw = NFW()
        self.tnfw = TNFW()

    def test_deflction(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        xdef_t, ydef_t = self.tnfw.derivatives(x, y, Rs, theta_Rs, r_trunc)
        xdef, ydef = self.nfw.derivatives(x, y, Rs, theta_Rs)

        np.testing.assert_almost_equal(xdef_t, xdef, 5)
        np.testing.assert_almost_equal(ydef_t, ydef, 5)

    def test_potential(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        pot_t = self.tnfw.nfwPot((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc)
        pot = self.nfw.nfwPot((x ** 2 + y ** 2) ** .5, Rs, theta_Rs)

        np.testing.assert_almost_equal(pot, pot_t, 4)

    def test_gamma(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        g1t, g2t = self.tnfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, r_trunc, x, y)
        g1, g2 = self.nfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, theta_Rs, x, y)

        np.testing.assert_almost_equal(g1t, g1, 2)
        np.testing.assert_almost_equal(g2t, g2, 2)

    def test_hessian(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 100)
        y = np.linspace(0.2, 1, 100)

        xxt, yyt, xyt = self.tnfw.hessian(x, y, Rs, theta_Rs, r_trunc)
        xx, yy, xy = self.nfw.hessian(x, y, Rs, theta_Rs)
        print((xx - xxt)/xxt, 'test')
        #np.testing.assert_almost_equal(xy, xyt, 4)
        #np.testing.assert_almost_equal(yy, yyt, 4)
        #np.testing.assert_almost_equal(xy, xyt, 4)


if __name__ == '__main__':
    pytest.main()

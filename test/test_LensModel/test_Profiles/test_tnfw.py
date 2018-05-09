__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.nfw import NFW

import numpy as np
import numpy.testing as npt
import pytest


class TestTNFW(object):

    def setup(self):
        self.nfw = NFW()
        self.tnfw = TNFW()

    def test_deflection(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        xdef_t, ydef_t = self.tnfw.derivatives(x, y, Rs, theta_Rs, r_trunc)
        xdef, ydef = self.nfw.derivatives(x, y, Rs, theta_Rs)

        np.testing.assert_almost_equal(xdef_t, xdef, 5)
        np.testing.assert_almost_equal(ydef_t, ydef, 5)
        f_x_t, f_y_t = self.tnfw.derivatives(1., 0, Rs, theta_Rs, r_trunc=1.)
        npt.assert_almost_equal(f_x_t, 0.01731384025307516, decimal=5)

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

        np.testing.assert_almost_equal(g1t, g1, 5)
        np.testing.assert_almost_equal(g2t, g2, 5)

    def test_hessian(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 100)
        y = np.linspace(0.2, 1, 100)

        xxt, yyt, xyt = self.tnfw.hessian(x, y, Rs, theta_Rs, r_trunc)
        xx, yy, xy = self.nfw.hessian(x, y, Rs, theta_Rs)
        print((xx - xxt)/xxt, 'test')
        np.testing.assert_almost_equal(xy, xyt, 4)
        np.testing.assert_almost_equal(yy, yyt, 4)
        np.testing.assert_almost_equal(xy, xyt, 4)

    def test_density_2d(self):
        Rs = 0.2
        theta_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 3 * Rs, 1000)
        y = np.linspace(0.2, 0.5, 1000)

        kappa_t = self.tnfw.density_2d(x, y, Rs, theta_Rs, r_trunc)
        kappa = self.nfw.density_2d(x, y, Rs, theta_Rs)
        np.testing.assert_almost_equal(kappa, kappa_t, 5)


if __name__ == '__main__':
    pytest.main()

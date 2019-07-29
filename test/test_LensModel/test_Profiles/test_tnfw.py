__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest


class TestTNFW(object):

    def setup(self):
        self.nfw = NFW()
        self.tnfw = TNFW()

    def test_deflection(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.0 * Rs, 5 * Rs, 1000)
        y = np.linspace(0., 1, 1000)

        xdef_t, ydef_t = self.tnfw.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        xdef, ydef = self.nfw.derivatives(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(xdef_t, xdef, 5)
        np.testing.assert_almost_equal(ydef_t, ydef, 5)

    def test_potential(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        pot_t = self.tnfw.function(x, y, Rs, alpha_Rs, r_trunc)
        pot = self.nfw.function(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(pot, pot_t, 4)

        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs

        x = np.linspace(0.1, 0.7, 100)

        pot1 = self.tnfw.function(x, 0, Rs, alpha_Rs, r_trunc)
        pot_nfw1 = self.nfw.function(x, 0, Rs, alpha_Rs)
        npt.assert_almost_equal(pot1, pot_nfw1, 5)

    def test_gamma(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        g1t, g2t = self.tnfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, alpha_Rs, r_trunc, x, y)
        g1, g2 = self.nfw.nfwGamma((x ** 2 + y ** 2) ** .5, Rs, alpha_Rs, x, y)

        np.testing.assert_almost_equal(g1t, g1, 5)
        np.testing.assert_almost_equal(g2t, g2, 5)

    def test_hessian(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 100)
        y = np.linspace(0.2, 1, 100)

        xxt, yyt, xyt = self.tnfw.hessian(x, y, Rs, alpha_Rs, r_trunc)
        xx, yy, xy = self.nfw.hessian(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(xy, xyt, 4)
        np.testing.assert_almost_equal(yy, yyt, 4)
        np.testing.assert_almost_equal(xy, xyt, 4)

        Rs = 0.2
        r_trunc = 5
        xxt, yyt, xyt = self.tnfw.hessian(Rs, 0, Rs, alpha_Rs, r_trunc)
        xxt_delta, yyt_delta, xyt_delta = self.tnfw.hessian(Rs+0.000001, 0, Rs, alpha_Rs, r_trunc)
        npt.assert_almost_equal(xxt, xxt_delta, decimal=6)

    def test_density_2d(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 3 * Rs, 1000)
        y = np.linspace(0.2, 0.5, 1000)

        kappa_t = self.tnfw.density_2d(x, y, Rs, alpha_Rs, r_trunc)
        kappa = self.nfw.density_2d(x, y, Rs, alpha_Rs)
        np.testing.assert_almost_equal(kappa, kappa_t, 5)

    def test_transform(self):

        rho0, Rs = 1, 2

        trs = self.tnfw._rho02alpha(rho0, Rs)
        rho_out = self.tnfw._alpha2rho0(trs, Rs)

        npt.assert_almost_equal(rho0, rho_out)

    def test_numerical_derivatives(self):

        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1.5 * Rs

        diff = 1e-9

        x0, y0 = 0.1, 0.1

        x_def_t, y_def_t = self.tnfw.derivatives(x0,y0,Rs,alpha_Rs,r_trunc)
        x_def_t_deltax, _ = self.tnfw.derivatives(x0+diff, y0, Rs, alpha_Rs,r_trunc)
        x_def_t_deltay, y_def_t_deltay = self.tnfw.derivatives(x0, y0 + diff, Rs, alpha_Rs,r_trunc)
        actual = self.tnfw.hessian(x0,y0,Rs,alpha_Rs,r_trunc)

        f_xx_approx = (x_def_t_deltax - x_def_t) * diff ** -1
        f_yy_approx = (y_def_t_deltay - y_def_t) * diff ** -1
        f_xy_approx = (x_def_t_deltay - y_def_t) * diff ** -1
        numerical = [f_xx_approx,f_yy_approx,f_xy_approx]

        for (approx,true) in zip(numerical,actual):
            npt.assert_almost_equal(approx,true)


if __name__ == '__main__':
    pytest.main()

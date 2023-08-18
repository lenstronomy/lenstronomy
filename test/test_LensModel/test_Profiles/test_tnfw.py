__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.tnfw import TNFW
from lenstronomy.LensModel.Profiles.nfw import NFW
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest


class TestTNFW(object):
    def setup_method(self):
        self.nfw = NFW()
        self.tnfw = TNFW()

    def test_deflection(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.0 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.0, 1, 1000)

        xdef_t, ydef_t = self.tnfw.derivatives(x, y, Rs, alpha_Rs, r_trunc)
        xdef, ydef = self.nfw.derivatives(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(xdef_t, xdef, 5)
        np.testing.assert_almost_equal(ydef_t, ydef, 5)

    def test_potential_limit(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        pot_t = self.tnfw.function(x, y, Rs, alpha_Rs, r_trunc)
        pot = self.nfw.function(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(pot, pot_t, 4)

    def test_potential_derivative(self):
        Rs = 1.2
        alpha_Rs = 1
        r_trunc = 3 * Rs
        R = np.linspace(0.5 * Rs, 2.2 * Rs, 5000)
        dx = R[1] - R[0]

        alpha_tnfw = self.tnfw.nfwAlpha(R, Rs, 1, r_trunc, R, 0)[0]

        potential_array = self.tnfw.nfwPot(R, Rs, 1, r_trunc)
        alpha_tnfw_num_array = np.gradient(potential_array, dx)

        potential_from_float = [self.tnfw.nfwPot(R_i, Rs, 1, r_trunc) for R_i in R]
        alpha_tnfw_num_from_float = np.gradient(potential_from_float, dx)

        npt.assert_almost_equal(alpha_tnfw_num_array, alpha_tnfw, 4)
        npt.assert_almost_equal(alpha_tnfw_num_from_float, alpha_tnfw, 4)

    def test_gamma(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 1000)
        y = np.linspace(0.2, 1, 1000)

        g1t, g2t = self.tnfw.nfwGamma(
            (x**2 + y**2) ** 0.5, Rs, alpha_Rs, r_trunc, x, y
        )
        g1, g2 = self.nfw.nfwGamma((x**2 + y**2) ** 0.5, Rs, alpha_Rs, x, y)

        np.testing.assert_almost_equal(g1t, g1, 5)
        np.testing.assert_almost_equal(g2t, g2, 5)

    def test_hessian(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1000000000000 * Rs
        x = np.linspace(0.1 * Rs, 5 * Rs, 100)
        y = np.linspace(0.2, 1, 100)

        xxt, xyt, yxt, yyt = self.tnfw.hessian(x, y, Rs, alpha_Rs, r_trunc)
        xx, xy, yx, yy = self.nfw.hessian(x, y, Rs, alpha_Rs)

        np.testing.assert_almost_equal(xy, xyt, 4)
        np.testing.assert_almost_equal(yy, yyt, 4)
        np.testing.assert_almost_equal(xy, xyt, 4)
        np.testing.assert_almost_equal(yxt, xyt, 8)

        Rs = 0.2
        r_trunc = 5
        xxt, xyt, yxt, yyt = self.tnfw.hessian(Rs, 0, Rs, alpha_Rs, r_trunc)
        xxt_delta, xyt_delta, yxt_delta, yyt_delta = self.tnfw.hessian(
            Rs + 0.000001, 0, Rs, alpha_Rs, r_trunc
        )
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

        trs = self.tnfw.rho02alpha(rho0, Rs)
        rho_out = self.tnfw.alpha2rho0(trs, Rs)

        npt.assert_almost_equal(rho0, rho_out)

    def test_numerical_derivatives(self):
        Rs = 0.2
        alpha_Rs = 0.1
        r_trunc = 1.5 * Rs

        diff = 1e-9

        x0, y0 = 0.1, 0.1

        x_def_t, y_def_t = self.tnfw.derivatives(x0, y0, Rs, alpha_Rs, r_trunc)
        x_def_t_deltax, _ = self.tnfw.derivatives(x0 + diff, y0, Rs, alpha_Rs, r_trunc)
        x_def_t_deltay, y_def_t_deltay = self.tnfw.derivatives(
            x0, y0 + diff, Rs, alpha_Rs, r_trunc
        )
        actual = self.tnfw.hessian(x0, y0, Rs, alpha_Rs, r_trunc)

        f_xx_approx = (x_def_t_deltax - x_def_t) * diff**-1
        f_yy_approx = (y_def_t_deltay - y_def_t) * diff**-1
        f_xy_approx = (x_def_t_deltay - y_def_t) * diff**-1
        numerical = [f_xx_approx, f_xy_approx, f_xy_approx, f_yy_approx]

        for approx, true in zip(numerical, actual):
            npt.assert_almost_equal(approx, true)

    def test_F_function_at_one(self):
        f_tnfw = self.tnfw.F(1.0)
        npt.assert_(f_tnfw == 1)
        f_tnfw = self.tnfw.F(np.ones((2, 2)))
        f_tnfw = f_tnfw.ravel()
        for value in f_tnfw:
            npt.assert_(value == 1)

    def test_F_function_at_zero(self):
        f_tnfw = self.tnfw.F(0)
        npt.assert_almost_equal(f_tnfw, 0, decimal=8)

    def test__cos_function(self):
        # test private _cos_function function for raise
        x = 3 + 6j
        npt.assert_raises(Exception, self.tnfw._cos_function, x)


if __name__ == "__main__":
    pytest.main()

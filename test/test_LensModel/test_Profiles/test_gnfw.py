__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.gnfw import GNFW

import numpy as np
import numpy.testing as npt
import pytest


class TestGNFW(object):
    """This class tests the generalized NFW profile."""

    def setup_method(self):
        self.nfw = NFW()
        self.gnfw = GNFW()
        self.gnfw_trapezoidal = GNFW(trapezoidal_integration=True)

    def test_function(self):
        """Tests `GNFW.function()`"""
        x = 1
        y = 2
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        alpha_Rs = self.nfw.rho02alpha(rho0, Rs)
        values_nfw = self.nfw.function(x, y, Rs, alpha_Rs)

        kappa_s = self.gnfw.rho02kappa_s(rho0, Rs, gamma_in)
        values_gnfw = self.gnfw.function(x, y, Rs, kappa_s, gamma_in)
        npt.assert_almost_equal(values_nfw, values_gnfw, decimal=5)

        values_gnfw_trapezoidal = self.gnfw_trapezoidal.function(
            x, y, Rs, kappa_s, gamma_in
        )
        npt.assert_almost_equal(values_nfw, values_gnfw_trapezoidal, decimal=4)

        # test for array of values
        x = np.linspace(0.5, 10, 10)
        y = np.ones_like(x)
        values_nfw = self.nfw.function(x, y, Rs, alpha_Rs)
        values_gnfw = self.gnfw.function(x, y, Rs, kappa_s, gamma_in)
        npt.assert_almost_equal(values_nfw, values_gnfw, decimal=5)

    def test_derivatives(self):
        """Tests `GNFW.derivatives()`"""
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        alpha_Rs = self.nfw.rho02alpha(rho0, Rs)
        f_x_nfw, f_y_nfw = self.nfw.derivatives(x, y, Rs, alpha_Rs)

        kappa_s = self.gnfw.rho02kappa_s(rho0, Rs, gamma_in)
        f_x_gnfw, f_y_gnfw = self.gnfw.derivatives(x, y, Rs, kappa_s, gamma_in)

        npt.assert_almost_equal(f_x_nfw, f_x_gnfw, decimal=10)
        npt.assert_almost_equal(f_y_nfw, f_y_gnfw, decimal=10)

        f_x_gnfwt, f_y_gnfwt = self.gnfw_trapezoidal.derivatives(
            x, y, Rs, kappa_s, gamma_in
        )
        npt.assert_almost_equal(f_x_nfw, f_x_gnfwt, decimal=5)
        npt.assert_almost_equal(f_y_nfw, f_y_gnfwt, decimal=5)

        # test for really small Rs
        Rs = 0.00000001
        f_x_nfw, f_y_nfw = self.nfw.derivatives(x, y, Rs, alpha_Rs)
        f_x_gnfw, f_y_gnfw = self.gnfw.derivatives(x, y, Rs, kappa_s, gamma_in)
        npt.assert_almost_equal(f_x_nfw, f_x_gnfw, decimal=3)
        npt.assert_almost_equal(f_y_nfw, f_y_gnfw, decimal=3)

    def test_hessian(self):
        """Tests `GNFW.hessian()`"""
        x = np.linspace(0.5, 10, 10)
        y = x * 0.0
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        alpha_Rs = self.nfw.rho02alpha(rho0, Rs)
        f_xx_nfw, f_xy_nfw, _, f_yy_nfw = self.nfw.hessian(x, y, Rs, alpha_Rs)

        kappa_s = self.gnfw.rho02kappa_s(rho0, Rs, gamma_in)
        f_xx_gnfw, f_xy_gnfw, _, f_yy_gnfw = self.gnfw.hessian(
            x, y, Rs, kappa_s, gamma_in
        )

        npt.assert_almost_equal(f_xx_nfw, f_xx_gnfw, decimal=10)
        npt.assert_almost_equal(f_yy_nfw, f_yy_gnfw, decimal=10)
        npt.assert_almost_equal(f_xy_nfw, f_xy_gnfw, decimal=10)

        f_xx_gnfwt, f_xy_gnfwt, _, f_yy_gnfwt = self.gnfw_trapezoidal.hessian(
            x, y, Rs, kappa_s, gamma_in
        )
        npt.assert_almost_equal(f_xx_nfw, f_xx_gnfwt, decimal=4)
        npt.assert_almost_equal(f_yy_nfw, f_yy_gnfwt, decimal=4)
        npt.assert_almost_equal(f_xy_nfw, f_xy_gnfwt, decimal=4)

        # test for really small Rs
        Rs = 0.00000001
        f_xx_nfw, f_xy_nfw, _, f_yy_nfw = self.nfw.hessian(x, y, Rs, alpha_Rs)
        f_xx_gnfw, f_xy_gnfw, _, f_yy_gnfw = self.gnfw.hessian(
            x, y, Rs, kappa_s, gamma_in
        )

        npt.assert_almost_equal(f_xx_nfw, f_xx_gnfw, decimal=2)
        npt.assert_almost_equal(f_yy_nfw, f_yy_gnfw, decimal=2)
        npt.assert_almost_equal(f_xy_nfw, f_xy_gnfw, decimal=2)

    def test_density(self):
        """Tests `GNFW.density()`"""
        R = 1
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        density_nfw = self.nfw.density(R, Rs, rho0)
        density_gnfw = self.gnfw.density(R, Rs, rho0, gamma_in)
        density_gnfwt = self.gnfw_trapezoidal.density(R, Rs, rho0, gamma_in)

        npt.assert_almost_equal(density_nfw, density_gnfw, decimal=10)
        npt.assert_almost_equal(density_nfw, density_gnfwt, decimal=6)

    def test_density_lens(self):
        """Tests `GNFW.density_lens()`"""
        R = 1
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        alpha_Rs = self.nfw.rho02alpha(rho0, Rs)
        density_nfw = self.nfw.density_lens(R, Rs, alpha_Rs)

        kappa_s = self.gnfw.rho02kappa_s(rho0, Rs, gamma_in)
        density_gnfw = self.gnfw.density_lens(R, Rs, kappa_s, gamma_in)
        density_gnfwt = self.gnfw_trapezoidal.density_lens(R, Rs, kappa_s, gamma_in)

        npt.assert_almost_equal(density_nfw, density_gnfw, decimal=10)
        npt.assert_almost_equal(density_nfw, density_gnfwt, decimal=6)

    def test_density_2d(self):
        """Tests `GNFW.density_2d_lens()`"""
        x = np.array([1])
        y = np.array([2])
        Rs = 1.0
        rho0 = 10
        gamma_in = 1

        kappa_nfw = self.nfw.density_2d(x, y, Rs, rho0)
        kappa_gnfw = self.gnfw.density_2d(x, y, Rs, rho0, gamma_in)
        kappa_gnfwt = self.gnfw_trapezoidal.density_2d(x, y, Rs, rho0, gamma_in)
        npt.assert_almost_equal(kappa_nfw, kappa_gnfw, decimal=10)
        npt.assert_almost_equal(kappa_nfw, kappa_gnfwt, decimal=5)

    def test_mass_3d(self):
        """Tests `GNFW.mass_3d()`"""
        r = 1
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        mass_3d_nfw = self.nfw.mass_3d(r, Rs, rho0)
        mass_3d_gnfw = self.gnfw.mass_3d(r, Rs, rho0, gamma_in)

        npt.assert_almost_equal(mass_3d_nfw, mass_3d_gnfw, decimal=10)

    def test_mass_3d_lens(self):
        """Tests `GNFW.mass_3d_lens()`"""
        r = 1
        Rs = 1.0
        rho0 = 1
        gamma_in = 1

        alpha_Rs = self.nfw.rho02alpha(rho0, Rs)
        mass_3d_nfw = self.nfw.mass_3d_lens(r, Rs, alpha_Rs)

        kappa_s = self.gnfw.rho02kappa_s(rho0, Rs, gamma_in)
        mass_3d_gnfw = self.gnfw.mass_3d_lens(r, Rs, kappa_s, gamma_in)

        npt.assert_almost_equal(mass_3d_nfw, mass_3d_gnfw, decimal=10)


if __name__ == "__main__":
    pytest.main()

__author__ = "lynevdv"


from lenstronomy.LensModel.Profiles.multipole import Multipole, EllipticalMultipole
from lenstronomy.Util import util

import numpy as np
import pytest
import numpy.testing as npt


class TestMultipole(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.Multipole = Multipole()

    def test_function(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values, 0.006684307, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        assert values[0] == 0

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values[0], -0.007409114, decimal=6)
        npt.assert_almost_equal(values[1], -0.009453038, decimal=6)
        npt.assert_almost_equal(values[2], -0.009910505, decimal=6)

    def test_derivatives(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x, -0.003939644, decimal=6)
        npt.assert_almost_equal(f_y, 0.005311976, decimal=6)

        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x[0], -0.003939644, decimal=6)
        npt.assert_almost_equal(f_y[0], 0.005311976, decimal=6)

        x = np.array([2, 3, 1])
        y = np.array([1, 1, 4])
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x[0], -0.003613858, decimal=6)
        npt.assert_almost_equal(f_x[1], -0.000970385, decimal=6)
        npt.assert_almost_equal(f_x[2], 0.005970704, decimal=6)
        npt.assert_almost_equal(f_y[0], -0.000181398, decimal=6)
        npt.assert_almost_equal(f_y[1], -0.006541883, decimal=6)
        npt.assert_almost_equal(f_y[2], 0.001649720, decimal=6)

    def test_hessian(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        f_xx, f_xy, f_yx, f_yy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx, -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy, -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy, 0.008021169, decimal=6)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        x = np.array([1])
        y = np.array([2])
        f_xx, f_xy, f_yx, f_yy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx[0], -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy[0], -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy[0], 0.008021169, decimal=6)
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values[0][0], -0.016042338, decimal=6)
        npt.assert_almost_equal(values[3][0], -0.004010584, decimal=6)
        npt.assert_almost_equal(values[1][0], 0.008021169, decimal=6)
        npt.assert_almost_equal(values[0][1], 0.001417956, decimal=6)
        npt.assert_almost_equal(values[3][1], 0.012761602, decimal=6)
        npt.assert_almost_equal(values[1][1], -0.004253867, decimal=6)


class TestEllipticalMultipole(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.Multipole = EllipticalMultipole()
        self.SphericalMultipole = Multipole()
        self.x, self.y = util.make_grid(numPix=10, deltapix=0.2)

    def test_function(self):
        x = 1
        y = 2
        q = 0.5
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        values = self.Multipole.function(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(values, 0.017025477, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.Multipole.function(x, y, m, a_m, phi_m, q)
        assert values[0] == 0

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.Multipole.function(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(values[0], 0.005762831, decimal=6)
        npt.assert_almost_equal(values[1], 0.000954611, decimal=6)
        npt.assert_almost_equal(values[2], -0.002321308, decimal=6)

        # Test that the limit q-> 1 is consistent with the spherical multipoles
        function_ell_limit = self.Multipole.function(
            self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24, q=0.99989
        )
        function_sph = self.SphericalMultipole.function(
            self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24
        )
        npt.assert_allclose(function_ell_limit, function_sph, rtol=1e-5, atol=5e-5)

        function_ell_limit = self.Multipole.function(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18, q=0.99989
        )
        function_sph = self.SphericalMultipole.function(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18
        )
        npt.assert_allclose(function_ell_limit, function_sph, rtol=1e-5, atol=5e-5)

    def test_derivatives(self):
        x = 1
        y = 2
        q = 0.5
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(f_x, 0.009253496, decimal=6)
        npt.assert_almost_equal(f_y, 0.003885991, decimal=6)

        x = np.array([2, 3, 1])
        y = np.array([1, 1, 4])
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(f_x[0], -0.005559822, decimal=6)
        npt.assert_almost_equal(f_x[1], -0.003963335, decimal=6)
        npt.assert_almost_equal(f_x[2], 0.015182005, decimal=6)
        npt.assert_almost_equal(f_y[0], 0.016882475, decimal=6)
        npt.assert_almost_equal(f_y[1], 0.012844618, decimal=6)
        npt.assert_almost_equal(f_y[2], 0.001639347, decimal=6)

        # Test that the limit q-> 1 is consistent with the spherical multipoles
        alpha_x_ell_limit, alpha_y_ell_limit = self.Multipole.derivatives(
            self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24, q=0.99989
        )
        alpha_x_sph, alpha_y_sph = self.SphericalMultipole.derivatives(
            self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24
        )
        npt.assert_allclose(alpha_x_ell_limit, alpha_x_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(alpha_y_ell_limit, alpha_y_sph, rtol=1e-5, atol=5e-5)

        alpha_x_ell_limit, alpha_y_ell_limit = self.Multipole.derivatives(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18, q=0.99989
        )
        alpha_x_sph, alpha_y_sph = self.SphericalMultipole.derivatives(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18
        )
        npt.assert_allclose(alpha_x_ell_limit, alpha_x_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(alpha_y_ell_limit, alpha_y_sph, rtol=1e-5, atol=5e-5)

    def test_hessian(self):
        x = 1
        y = 2
        q = 0.5
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.0
        f_xx, f_xy, f_yx, f_yy = self.Multipole.hessian(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(f_xx, -0.01254782, decimal=6)
        npt.assert_almost_equal(f_yy, -0.003136955, decimal=6)
        npt.assert_almost_equal(f_xy, 0.00627391, decimal=6)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.Multipole.hessian(x, y, m, a_m, phi_m, q)
        npt.assert_almost_equal(values[0][0], 0.000868241, decimal=6)
        npt.assert_almost_equal(values[0][1], 0.001611182, decimal=6)
        npt.assert_almost_equal(values[0][2], 0.000924536, decimal=6)
        npt.assert_almost_equal(values[3][0], 0.003472964, decimal=6)
        npt.assert_almost_equal(values[3][1], 0.014500636, decimal=6)
        npt.assert_almost_equal(values[3][2], 0.014792568, decimal=6)
        npt.assert_almost_equal(values[1][0], -0.001736482, decimal=6)
        npt.assert_almost_equal(values[1][1], -0.004833545, decimal=6)
        npt.assert_almost_equal(values[1][2], -0.003698142, decimal=6)

        # Test that the limit q-> 1 is consistent with the spherical multipoles
        f_xx_ell_limit, f_xy_ell_limit, f_yx_ell_limit, f_yy_ell_limit = (
            self.Multipole.hessian(
                self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24, q=0.99989
            )
        )
        f_xx_sph, f_xy_sph, f_yx_sph, f_yy_sph = self.SphericalMultipole.hessian(
            self.x, self.y, m=4, a_m=a_m, phi_m=np.pi / 24
        )
        npt.assert_allclose(f_xx_ell_limit, f_xx_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_xy_ell_limit, f_xy_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_yx_ell_limit, f_yx_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_yy_ell_limit, f_yy_sph, rtol=1e-5, atol=5e-5)

        f_xx_ell_limit, f_xy_ell_limit, _, f_yy_ell_limit = self.Multipole.hessian(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18, q=0.99989
        )
        f_xx_sph, f_xy_sph, _, f_yy_sph = self.SphericalMultipole.hessian(
            self.x, self.y, m=3, a_m=a_m, phi_m=np.pi / 18
        )
        npt.assert_allclose(f_xx_ell_limit, f_xx_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_xy_ell_limit, f_xy_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_yx_ell_limit, f_yx_sph, rtol=1e-5, atol=5e-5)
        npt.assert_allclose(f_yy_ell_limit, f_yy_sph, rtol=1e-5, atol=5e-5)


if __name__ == "__main__":
    pytest.main()

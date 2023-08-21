__author__ = "lynevdv"


from lenstronomy.LensModel.Profiles.elliptical_density_slice import ElliSLICE

import numpy as np
import pytest
import numpy.testing as npt


class TestElliSLICE(object):
    """Tests the elliptical slice lens model."""

    def setup_method(self):
        self.ElliSLICE = ElliSLICE()

    def test_function(self):
        x = 0.5
        y = 0.1
        a = 2.0
        b = 1.0
        psi = 30 * np.pi / 180.0
        sigma_0 = 5.0
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values, 4.532482297, decimal=4)

        x = 3.0 * np.sqrt(3) / 2.0
        y = 3.0 / 2.0
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values, 15.52885056, decimal=4)

        x = np.array([0])
        y = np.array([0])
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values[0], 4.054651081, decimal=5)

        x = np.array([np.sqrt(3), np.sqrt(3) + 0.000000001, np.sqrt(3) - 0.000000001])
        y = np.array([1, 1.000000001, 0.999999999])
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values[0], values[1], decimal=5)
        npt.assert_almost_equal(values[1], values[2], decimal=5)

    def test_derivatives(self):
        x = 0.5
        y = 0.1
        a = 2.0
        b = 1.0
        psi = 30 * np.pi / 180.0
        sigma_0 = 5.0
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x, 1.938995765, decimal=6)
        npt.assert_almost_equal(f_y, -0.13835403, decimal=6)

        x = 4
        y = 0.0
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, 0.0, sigma_0)
        npt.assert_almost_equal(f_x, 2.629658164, decimal=6)
        npt.assert_almost_equal(f_y, 0.0, decimal=6)

        x = np.array([0.5])
        y = np.array([0.1])
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x, 1.938995765, decimal=6)
        npt.assert_almost_equal(f_y, -0.13835403, decimal=6)

        x = np.array([np.sqrt(3), np.sqrt(3) + 0.000000001, np.sqrt(3) - 0.000000001])
        y = np.array([1, 1.000000001, 0.999999999])
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x[0], f_x[1], decimal=5)
        npt.assert_almost_equal(f_y[1], f_y[2], decimal=5)

    def test_hessian(self):
        x = 0.5
        y = 0.1
        a = 2.0
        b = 1.0
        psi = 30 * np.pi / 180.0
        sigma_0 = 5.0
        f_xx, f_xy, f_yx, f_yy = self.ElliSLICE.hessian(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal((f_xx + f_yy) / 2.0, 5.0, decimal=6)
        x = np.array([1])
        y = np.array([2])
        npt.assert_almost_equal(f_xy, f_yx, decimal=7)
        f_xx, f_xy, f_yx, f_yy = self.ElliSLICE.hessian(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal((f_xx + f_yy) / 2.0, 0.0, decimal=6)
        x = np.array([1, 3, 0.0])
        y = np.array([2, 1, 0.5])
        values = self.ElliSLICE.hessian(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal((values[0][2] + values[3][2]) / 2.0, 5.0, decimal=6)


if __name__ == "__main__":
    pytest.main()

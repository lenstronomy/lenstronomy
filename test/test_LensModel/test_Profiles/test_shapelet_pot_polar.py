__author__ = "sibirrer"

from lenstronomy.LensModel.Profiles.shapelet_pot_polar import PolarShapelets

import numpy as np
import numpy.testing as npt
import pytest


class TestCartShapelets(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.polarShapelets = PolarShapelets()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        beta = 1.0
        coeffs = (1.0, 1.0)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values[0], -0.046311501189135587, decimal=8)

        x = 1.0
        y = 2.0
        beta = 1.0
        coeffs = (1.0, 1.0)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values, -0.046311501189135587, decimal=8)

        x = np.array([0])
        y = np.array([0])
        beta = 1.0
        coeffs = (0, 1.0)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values[0], 0, decimal=8)

        coeffs = (1, 1.0, 0, 0, 1, 1)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values[0], 0, decimal=8)

        coeffs = (1, 1.0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values[0], 0, decimal=8)

        coeffs = (0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        values = self.polarShapelets.function(x, y, coeffs, beta)
        npt.assert_almost_equal(values[0], 0, decimal=8)

    def test_derivatives(self):
        """

        :return:
        """
        beta = 1.0
        coeffs = [1, 0, 0, 1.0, 0, 0, 0, 0]
        kwargs_lens1 = {"coeffs": coeffs, "beta": beta}

        x1 = 1.0
        y1 = 2.0
        f_x1, f_y1 = self.polarShapelets.derivatives(x1, y1, **kwargs_lens1)
        x2 = np.array([1.0])
        y2 = np.array([2.0])
        f_x2, f_y2 = self.polarShapelets.derivatives(x2, y2, **kwargs_lens1)
        assert f_x1 == f_x2[0]
        npt.assert_almost_equal(f_x1, -0.046311501189135601, decimal=8)
        npt.assert_almost_equal(f_y1, -0.092623002378271174, decimal=8)

        x3 = np.array([1.0, 0])
        y3 = np.array([2.0, 0])
        f_x3, f_y3 = self.polarShapelets.derivatives(x3, y3, **kwargs_lens1)
        assert f_x1 == f_x3[0]

    def test_hessian(self):
        beta = 1.0
        coeffs = [1, 1, 0, 1.0, 0, 0, 0, 0]
        kwargs_lens1 = {"coeffs": coeffs, "beta": beta}

        x1 = np.array([1.0, 2])
        y1 = np.array([1, 1])
        f_xx, f_xy, f_yx, f_yy = self.polarShapelets.hessian(x1, y1, **kwargs_lens1)
        npt.assert_almost_equal(f_xx[0], 0.20755374871029733, decimal=8)
        npt.assert_almost_equal(f_yy[0], 0.20755374871029728, decimal=8)
        npt.assert_almost_equal(f_xy[0], -0.20755374871029739, decimal=8)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)


if __name__ == "__main__":
    pytest.main()

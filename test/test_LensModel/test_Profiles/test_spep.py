__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.spep import SPEP
from lenstronomy.LensModel.Profiles.sie import SIE
import lenstronomy.Util.param_util as param_util

import numpy as np
import pytest
import numpy.testing as npt


class TestSPEP(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.SPEP = SPEP()
        self.SIE = SIE()

    def test_function(self):
        x = 1
        y = 2
        phi_E = 1.0
        gamma = 1.9
        q = 0.9
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.SPEP.function(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(values, 2.104213947346917, decimal=7)
        x = np.array([0])
        y = np.array([0])
        values = self.SPEP.function(x, y, phi_E, gamma, e1, e2)
        assert values[0] == 0

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.SPEP.function(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(values[0], 2.1709510681181285, decimal=7)
        npt.assert_almost_equal(values[1], 3.2293397784259108, decimal=7)
        npt.assert_almost_equal(values[2], 4.3624056004556948, decimal=7)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        gamma = 1.9
        q = 0.9
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x[0], 0.43989645846696634, decimal=7)
        npt.assert_almost_equal(f_y[0], 0.93736944180732129, decimal=7)

        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(values[0][0], 0.43989645846696634, decimal=7)
        npt.assert_almost_equal(values[1][0], 0.93736944180732129, decimal=7)
        npt.assert_almost_equal(values[0][1], 1.1029501948308649, decimal=7)
        npt.assert_almost_equal(values[1][1], 0.24342317177590794, decimal=7)

        x = 1
        y = 2
        phi_E = 1.0
        gamma = 1.9
        q = 0.9
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, 0.43989645846696634, decimal=7)
        npt.assert_almost_equal(f_y, 0.93736944180732129, decimal=7)
        x = 0
        y = 0
        f_x, f_y = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        assert f_x == 0
        assert f_y == 0

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        gamma = 1.9
        q = 0.9
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.SPEP.hessian(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_xx[0], 0.46312881977317422, decimal=7)
        npt.assert_almost_equal(f_yy[0], 0.15165326557198552, decimal=7)
        npt.assert_almost_equal(f_xy[0], -0.20956958696323871, decimal=7)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.SPEP.hessian(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(values[0][0], 0.46312881977317422, decimal=7)
        npt.assert_almost_equal(values[3][0], 0.15165326557198552, decimal=7)
        npt.assert_almost_equal(values[1][0], -0.20956958696323871, decimal=7)
        npt.assert_almost_equal(values[0][1], 0.070999592014527796, decimal=7)
        npt.assert_almost_equal(values[3][1], 0.33245358685908111, decimal=7)
        npt.assert_almost_equal(values[1][1], -0.10270375656049677, decimal=7)

    def test_spep_sie_conventions(self):
        x = np.array([1.0, 2.0, 0.0])
        y = np.array([2, 1.0, 1.0])
        phi_E = 1.0
        gamma = 2
        q = 0.9999
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.SPEP.hessian(x, y, phi_E, gamma, e1, e2)
        f_xx_sie, f_xy_sie, f_yx_sie, f_yy_sie = self.SIE.hessian(x, y, phi_E, e1, e2)
        npt.assert_almost_equal(f_xx, f_xx_sie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_sie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_sie, decimal=4)
        npt.assert_almost_equal(f_yx, f_yx_sie, decimal=4)


if __name__ == "__main__":
    pytest.main()

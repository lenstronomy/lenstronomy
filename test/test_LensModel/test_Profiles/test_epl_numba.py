__author__ = "sibirrer"


import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util


class TestEPL_numba(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        from lenstronomy.LensModel.Profiles.epl import EPL

        self.EPL = EPL()
        from lenstronomy.LensModel.Profiles.epl_numba import EPL_numba

        self.EPL_numba = EPL_numba()

    def test_function(self):
        phi_E = 1.0
        gamma = 2.0
        q = 0.999
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1.0, 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
        values_nb = self.EPL_numba.function(x, y, phi_E, gamma, e1, e2)
        delta_f = values[0] - values[1]
        delta_f_nb = values_nb[0] - values_nb[1]
        npt.assert_almost_equal(delta_f, delta_f_nb, decimal=10)

        q = 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1.0, 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
        values_nb = self.EPL_numba.function(x, y, phi_E, gamma, e1, e2)
        delta_f = values[0] - values[1]
        delta_f_nb = values_nb[0] - values_nb[1]
        npt.assert_almost_equal(delta_f, delta_f_nb, decimal=10)

        q = 0.4
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1.0, 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
        values_nb = self.EPL_numba.function(x, y, phi_E, gamma, e1, e2)
        delta_f = values[0] - values[1]
        delta_f_nb = values_nb[0] - values_nb[1]
        npt.assert_almost_equal(delta_f, delta_f_nb, decimal=10)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.0
        gamma = 1.8
        q = 1.0
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
        f_x_nb, f_y_nb = self.EPL_numba.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, f_x_nb, decimal=10)
        npt.assert_almost_equal(f_y, f_y_nb, decimal=10)

        q = 0.7
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
        f_x_nb, f_y_nb = self.EPL_numba.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, f_x_nb, decimal=10)
        npt.assert_almost_equal(f_y, f_y_nb, decimal=10)

    def test_hessian(self):
        x = np.array([1.0])
        y = np.array([2.0])
        phi_E = 1.0
        gamma = 2.2
        q = 0.9
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.EPL.hessian(x, y, phi_E, gamma, e1, e2)
        f_xx_nb, f_xy_nb, f_yx_nb, f_yy_nb = self.EPL_numba.hessian(
            x, y, phi_E, gamma, e1, e2
        )
        npt.assert_almost_equal(f_xx, f_xx_nb, decimal=10)
        npt.assert_almost_equal(f_yy, f_yy_nb, decimal=10)
        npt.assert_almost_equal(f_xy, f_xy_nb, decimal=10)

    def test_regularization(self):
        phi_E = 1.0
        gamma = 2.0
        q = 1.0
        phi_G = 1.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        x = 0.0
        y = 0.0
        f_x, f_y = self.EPL_numba.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, 0.0)
        npt.assert_almost_equal(f_y, 0.0)

        x = 0.0
        y = 0.0
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, 0.0)
        npt.assert_almost_equal(f_y, 0.0)

        x = 0.0
        y = 0.0
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma + 0.1, e1, e2)
        npt.assert_almost_equal(f_x, 0.0)
        npt.assert_almost_equal(f_y, 0.0)

        x = 0.0
        y = 0.0
        f = self.EPL_numba.function(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f, 0.0)

        x = 0.0
        y = 0.0
        f_xx, f_xy, f_yx, f_yy = self.EPL_numba.hessian(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_xx, 1e10, decimal=10)
        npt.assert_almost_equal(f_yy, 0, decimal=10)
        npt.assert_almost_equal(
            f_xy, 0, decimal=5
        )  # floating point cancellation, so less precise
        # Magnification:
        npt.assert_almost_equal(
            1 / ((1 - f_xx) * (1 - f_yy) - f_xy**2), 0.0, decimal=10
        )


if __name__ == "__main__":
    pytest.main()

__author__ = "dangilman"

import numpy as np
import numpy.testing as npt
import pytest
import copy
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
    EPL_MULTIPOLE_M1M3M4,
    EPL_MULTIPOLE_M1M3M4_ELL,
)
from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import (
    EPL_MULTIPOLE_M3M4,
    EPL_MULTIPOLE_M3M4_ELL,
)


class TestEPL_MULTIPOLE_M1M3M4(object):
    """Test EPL_MULTIPOLE_M1M3M4 vs EPL_MULTIPOLE_M3M4 with m1 term = 0.

    And test that EPL_MULTIPOLE_M1M3M4 and EPL_MULTIPOLE_M1M3M4_ELL are equivalent when
    q->1
    """

    def setup_method(self):

        self.epl_m1m3m4 = EPL_MULTIPOLE_M1M3M4()
        self.epl_m1m3m4_ell = EPL_MULTIPOLE_M1M3M4_ELL()
        self.epl_m3m4_ell = EPL_MULTIPOLE_M3M4_ELL()
        self.epl_m3m4 = EPL_MULTIPOLE_M3M4()
        self.kwargs_m1m3m4 = {
            "theta_E": 1.4,
            "center_x": 0.05,
            "center_y": -0.02,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a1_a": 0.0,
            "delta_phi_m1": 0.2,
            "a3_a": 0.05,
            "delta_phi_m3": 0.1,
            "a4_a": -0.05,
            "delta_phi_m4": 0.2,
        }
        self.kwargs_m3m4 = {
            "theta_E": 1.4,
            "center_x": 0.05,
            "center_y": -0.02,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a3_a": 0.05,
            "delta_phi_m3": 0.1,
            "a4_a": -0.05,
            "delta_phi_m4": 0.2,
        }

        e1, e2 = param_util.phi_q2_ellipticity(np.pi / 10, 0.9995)
        self.kwargs_m1m3m4_almostcirc = {
            "theta_E": 1.4,
            "center_x": 0.05,
            "center_y": -0.02,
            "e1": e1,
            "e2": e2,
            "gamma": 2.0,
            "a1_a": 0.04,
            "delta_phi_m1": 0.2,
            "a3_a": 0.05,
            "delta_phi_m3": 0.1,
            "a4_a": -0.05,
            "delta_phi_m4": 0.2,
        }
        self.x, self.y = util.make_grid(num_pix=10, delta_pix=0.2)

    def test_function(self):

        result_m1m3m4 = self.epl_m1m3m4.function(self.x, self.y, **self.kwargs_m1m3m4)
        result_m3m4 = self.epl_m3m4.function(self.x, self.y, **self.kwargs_m3m4)
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)

        ## Test that the limit q-> 1 is consistent between circular and elliptical multipoles
        result_m1m3m4_circ = self.epl_m1m3m4.function(
            self.x, self.y, **self.kwargs_m1m3m4_almostcirc
        )
        result_m1m3m4_ell_limit = self.epl_m1m3m4_ell.function(
            self.x, self.y, **self.kwargs_m1m3m4_almostcirc
        )
        npt.assert_allclose(
            result_m1m3m4_circ, result_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )

    def test_derivatives(self):

        alpha_x_m1m3m4, alpha_y_m1m3m4 = self.epl_m1m3m4.derivatives(
            self.x, self.y, **self.kwargs_m1m3m4
        )
        alpha_x_m3m4, alpha_y_m3m4 = self.epl_m3m4.derivatives(
            self.x, self.y, **self.kwargs_m3m4
        )
        npt.assert_almost_equal(alpha_x_m3m4, alpha_x_m1m3m4)
        npt.assert_almost_equal(alpha_y_m3m4, alpha_y_m1m3m4)

        ## Test that the limit q-> 1 is consistent between circular and elliptical multipoles
        alpha_x_m1m3m4_circ, alpha_y_m1m3m4_circ = self.epl_m1m3m4.derivatives(
            self.x, self.y, **self.kwargs_m1m3m4_almostcirc
        )
        alpha_x_m1m3m4_ell_limit, alpha_y_m1m3m4_ell_limit = (
            self.epl_m1m3m4_ell.derivatives(
                self.x, self.y, **self.kwargs_m1m3m4_almostcirc
            )
        )
        npt.assert_allclose(
            alpha_x_m1m3m4_circ, alpha_x_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )
        npt.assert_allclose(
            alpha_y_m1m3m4_circ, alpha_y_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )

    def test_hessian(self):

        f_xx_m1m3m4, f_xy_m1m3m4, _, f_yy_m1m3m4 = self.epl_m1m3m4.hessian(
            self.x, self.y, **self.kwargs_m1m3m4
        )
        f_xx_m3m4, f_xy_m3m4, _, f_yy_m3m4 = self.epl_m3m4.hessian(
            self.x, self.y, **self.kwargs_m3m4
        )
        npt.assert_almost_equal(f_xx_m3m4, f_xx_m1m3m4)
        npt.assert_almost_equal(f_xy_m3m4, f_xy_m1m3m4)
        npt.assert_almost_equal(f_yy_m3m4, f_yy_m1m3m4)

        ## Test that the limit q-> 1 is consistent between circular and elliptical multipoles
        f_xx_m1m3m4_circ, f_xy_m1m3m4_circ, _, f_yy_m1m3m4_circ = (
            self.epl_m1m3m4.hessian(self.x, self.y, **self.kwargs_m1m3m4_almostcirc)
        )
        f_xx_m1m3m4_ell_limit, f_xy_m1m3m4_ell_limit, _, f_yy_m1m3m4_ell_limit = (
            self.epl_m1m3m4_ell.hessian(self.x, self.y, **self.kwargs_m1m3m4_almostcirc)
        )
        npt.assert_allclose(
            f_xx_m1m3m4_circ, f_xx_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )
        npt.assert_allclose(
            f_xy_m1m3m4_circ, f_xy_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )
        npt.assert_allclose(
            f_yy_m1m3m4_circ, f_yy_m1m3m4_ell_limit, rtol=1e-4, atol=5e-5
        )


if __name__ == "__main__":
    pytest.main()

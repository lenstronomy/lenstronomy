__author__ = "dangilman"

import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.epl_multipole_m1m3m4 import (
    EPL_MULTIPOLE_M1M3M4,
    EPL_MULTIPOLE_M1M3M4_ELL,
)
from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import (
    EPL_MULTIPOLE_M3M4,
    EPL_MULTIPOLE_M3M4_ELL,
)


class TestEPL_MULTIPOLE_M3M4(object):
    """Test TestEPL_MULTIPOLE_M1M3M4 vs TestEPL_MULTIPOLE_M3M4 with m1 term = 0."""

    def setup_method(self):

        self.epl_m1m3m4 = EPL_MULTIPOLE_M1M3M4()
        self.epl_m1m3m4_ell = EPL_MULTIPOLE_M1M3M4_ELL()
        self.epl_m3m4_ell = EPL_MULTIPOLE_M3M4_ELL()
        self.epl_m3m4 = EPL_MULTIPOLE_M3M4()
        self.kwargs_m1m3m4 = {
            "theta_E": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a1_a": 0.0,
            "delta_phi_m1": 0.2,
            "a3_a": 0.05,
            "delta_phi_m3": 0.2,
            "a4_a": -0.05,
            "delta_phi_m4": 0.0,
        }
        self.kwargs_m3m4 = {
            "theta_E": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": -0.1,
            "gamma": 2.0,
            "a3_a": 0.05,
            "delta_phi_m3": 0.2,
            "a4_a": -0.05,
            "delta_phi_m4": 0.0,
        }

    def test_function(self):

        result_m1m3m4 = self.epl_m1m3m4.function(0.4, -0.2, **self.kwargs_m1m3m4)
        result_m3m4 = self.epl_m3m4.function(0.4, -0.2, **self.kwargs_m3m4)
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)
        result_m1m3m4 = self.epl_m1m3m4_ell.function(0.4, -0.2, **self.kwargs_m1m3m4)
        result_m3m4 = self.epl_m3m4_ell.function(0.4, -0.2, **self.kwargs_m3m4)
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)

    def test_derivatives(self):

        result_m1m3m4 = self.epl_m1m3m4.derivatives(0.4, -0.2, **self.kwargs_m1m3m4)[0]
        result_m3m4 = self.epl_m3m4.derivatives(0.4, -0.2, **self.kwargs_m3m4)[0]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)
        result_m1m3m4 = self.epl_m1m3m4_ell.derivatives(
            0.4, -0.2, **self.kwargs_m1m3m4
        )[0]
        result_m3m4 = self.epl_m3m4_ell.derivatives(0.4, -0.2, **self.kwargs_m3m4)[0]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)

        result_m1m3m4 = self.epl_m1m3m4.derivatives(0.4, -0.2, **self.kwargs_m1m3m4)[1]
        result_m3m4 = self.epl_m3m4.derivatives(0.4, -0.2, **self.kwargs_m3m4)[1]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)
        result_m1m3m4 = self.epl_m1m3m4_ell.derivatives(
            0.4, -0.2, **self.kwargs_m1m3m4
        )[1]
        result_m3m4 = self.epl_m3m4_ell.derivatives(0.4, -0.2, **self.kwargs_m3m4)[1]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)

    def test_hessian(self):

        result_m1m3m4 = self.epl_m1m3m4.hessian(0.4, -0.2, **self.kwargs_m1m3m4)[0]
        result_m3m4 = self.epl_m3m4.hessian(0.4, -0.2, **self.kwargs_m3m4)[0]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)
        result_m1m3m4 = self.epl_m1m3m4_ell.hessian(0.4, -0.2, **self.kwargs_m1m3m4)[0]
        result_m3m4 = self.epl_m3m4_ell.hessian(0.4, -0.2, **self.kwargs_m3m4)[0]
        npt.assert_almost_equal(result_m3m4, result_m1m3m4)


if __name__ == "__main__":
    pytest.main()

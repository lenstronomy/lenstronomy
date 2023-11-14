__author__ = "gipagano"


import numpy as np
import numpy.testing as npt
import pytest
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.nie_potential import NIE_POTENTIAL
from lenstronomy.LensModel.Profiles.spep import SPEP


class TestNIE_POTENTIAL(object):
    """Tests the NIE_POTENTIAL profile for different rotations."""

    def setup_method(self):
        self.nie_potential = NIE_POTENTIAL()
        self.spep = SPEP()

    def test_function(self):
        y = np.array([1.0, 2])
        x = np.array([0.0, 0.0])

        theta_E = 1.0
        theta_c = 0.0

        #############
        # no rotation
        #############

        e1, e2 = 0.05, 0.0
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the non-rotated output
        values = self.nie_potential.function(x, y, theta_E, theta_c, e1, e2)
        delta_pot = values[1] - values[0]
        values = self.spep.function(x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep)
        delta_pot_spep = values[1] - values[0]
        npt.assert_almost_equal(delta_pot, delta_pot_spep, decimal=4)

        ############
        # rotation 1
        ############

        e1, e2 = 0.05, 0.1
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        values = self.nie_potential.function(x, y, theta_E, theta_c, e1, e2)
        delta_pot = values[1] - values[0]
        values = self.spep.function(x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep)
        delta_pot_spep = values[1] - values[0]
        npt.assert_almost_equal(delta_pot, delta_pot_spep, decimal=4)

        ############
        # rotation 2
        ############

        e1, e2 = 0.15, 0.13
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        values = self.nie_potential.function(x, y, theta_E, theta_c, e1, e2)
        delta_pot = values[1] - values[0]
        values = self.spep.function(x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep)
        delta_pot_spep = values[1] - values[0]
        npt.assert_almost_equal(delta_pot, delta_pot_spep, decimal=4)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])

        theta_E = 1.0
        theta_c = 0.0

        #############
        # no rotation
        #############

        e1, e2 = 0.05, 0.0
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the non-rotated output
        f_x, f_y = self.nie_potential.derivatives(x, y, theta_E, theta_c, e1, e2)
        f_x_nie, f_y_nie = self.spep.derivatives(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

        ############
        # rotation 1
        ############

        e1, e2 = 0.05, 0.1
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        f_x, f_y = self.nie_potential.derivatives(x, y, theta_E, theta_c, e1, e2)
        f_x_nie, f_y_nie = self.spep.derivatives(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

        ############
        # rotation 2
        ############

        e1, e2 = 0.15, 0.13
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        f_x, f_y = self.nie_potential.derivatives(x, y, theta_E, theta_c, e1, e2)
        f_x_nie, f_y_nie = self.spep.derivatives(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])

        theta_E = 1.0
        theta_c = 0.0

        #############
        # no rotation
        #############

        e1, e2 = 0.05, 0.0
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the non-rotated output
        f_xx, f_xy, f_yx, f_yy = self.nie_potential.hessian(
            x, y, theta_E, theta_c, e1, e2
        )
        f_xx_nie, f_xy_nie, f_yx_nie, f_yy_nie = self.spep.hessian(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_xx, f_xx_nie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_nie, decimal=4)
        npt.assert_almost_equal(f_yx, f_yx_nie, decimal=4)

        ############
        # rotation 1
        ############

        e1, e2 = 0.05, 0.1
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        f_xx, f_xy, f_yx, f_yy = self.nie_potential.hessian(
            x, y, theta_E, theta_c, e1, e2
        )
        f_xx_nie, f_xy_nie, f_yx_nie, f_yy_nie = self.spep.hessian(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_xx, f_xx_nie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_nie, decimal=4)
        npt.assert_almost_equal(f_yx, f_yx_nie, decimal=4)

        ############
        # rotation 2
        ############

        e1, e2 = 0.15, 0.13
        eps = np.sqrt(e1**2 + e2**2)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)

        # map the nie_potential input to the spep input
        gamma_spep = 2.0
        q_spep = np.sqrt(q)
        e1_spep, e2_spep = param_util.phi_q2_ellipticity(phi_G, q_spep)
        theta_E_conv = self.nie_potential._theta_q_convert(theta_E, q)
        theta_E_spep = theta_E_conv * np.sqrt(1 - eps) / ((1 - eps) / (1 + eps)) ** 0.25

        # compare the rotated output
        f_xx, f_xy, f_yx, f_yy = self.nie_potential.hessian(
            x, y, theta_E, theta_c, e1, e2
        )
        f_xx_nie, f_xy_nie, f_yx_nie, f_yy_nie = self.spep.hessian(
            x, y, theta_E_spep, gamma_spep, e1_spep, e2_spep
        )
        npt.assert_almost_equal(f_xx, f_xx_nie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_nie, decimal=4)
        npt.assert_almost_equal(f_yx, f_yx_nie, decimal=4)

    def test_static(self):
        x, y = 1.0, 1.0
        phi_G, q = 0.3, 0.8

        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {"theta_E": 1.0, "theta_c": 0.1, "e1": e1, "e2": e2}
        f_ = self.nie_potential.function(x, y, **kwargs_lens)
        self.nie_potential.set_static(**kwargs_lens)
        f_static = self.nie_potential.function(x, y, **kwargs_lens)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        self.nie_potential.set_dynamic()
        kwargs_lens = {"theta_E": 2.0, "theta_c": 0.1, "e1": e1, "e2": e2}
        f_dyn = self.nie_potential.function(x, y, **kwargs_lens)
        assert f_dyn != f_static


if __name__ == "__main__":
    pytest.main()

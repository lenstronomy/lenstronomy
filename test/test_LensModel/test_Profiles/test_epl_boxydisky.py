__author__ = "Maverick-Oh"

import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util


class TestEPL_BOXYDISKY(object):
    """Test EPL_BOXYDISKY vs EPL + MULTIPOLE values."""

    def setup_method(self):
        from lenstronomy.LensModel.Profiles.epl import EPL

        self.epl = EPL()
        from lenstronomy.LensModel.Profiles.multipole import Multipole

        self.multipole = Multipole()
        from lenstronomy.LensModel.Profiles.epl_boxydisky import EPL_BOXYDISKY

        self.epl_boxydisky = EPL_BOXYDISKY()

        self.x, self.y = util.make_grid(numPix=10, deltapix=0.2)
        self.theta_E_list = [0.5, 1, 2]
        self.gamma_list = [1.8, 2.0, 2.2]
        self.e1_list = [-0.2, 0.0, 0.2]
        self.e2_list = [-0.2, 0.0, 0.2]
        self.a4_a_list = [0.0, 0.05, -0.05]

    def test_function(self):
        for gamma in self.gamma_list:
            for e1 in self.e1_list:
                for e2 in self.e2_list:
                    for theta_E in self.theta_E_list:
                        for a4_a in self.a4_a_list:
                            kwargs_epl = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                            }
                            phi, q = param_util.ellipticity2phi_q(e1, e2)
                            kwargs_multipole = {
                                "m": 4,
                                "a_m": a4_a * theta_E / np.sqrt(q),
                                "phi_m": phi,
                            }
                            kwargs_epl_boxydisky = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                                "a4_a": a4_a,
                            }
                            value1 = self.epl.function(
                                self.x, self.y, **kwargs_epl
                            ) + self.multipole.function(
                                self.x, self.y, **kwargs_multipole
                            )
                            value2 = self.epl_boxydisky.function(
                                self.x, self.y, **kwargs_epl_boxydisky
                            )
                            npt.assert_almost_equal(value1, value2, decimal=10)

    def test_derivatives(self):
        for gamma in self.gamma_list:
            for e1 in self.e1_list:
                for e2 in self.e2_list:
                    for theta_E in self.theta_E_list:
                        for a4_a in self.a4_a_list:
                            kwargs_epl = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                            }
                            phi, q = param_util.ellipticity2phi_q(e1, e2)
                            kwargs_multipole = {
                                "m": 4,
                                "a_m": a4_a * theta_E / np.sqrt(q),
                                "phi_m": phi,
                            }
                            kwargs_epl_boxydisky = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                                "a4_a": a4_a,
                            }

                            f_x1, f_y1 = self.epl.derivatives(
                                self.x, self.y, **kwargs_epl
                            )
                            f_x2, f_y2 = self.multipole.derivatives(
                                self.x, self.y, **kwargs_multipole
                            )
                            f_x = f_x1 + f_x2
                            f_y = f_y1 + f_y2
                            f_x_, f_y_ = self.epl_boxydisky.derivatives(
                                self.x, self.y, **kwargs_epl_boxydisky
                            )

                            npt.assert_almost_equal(f_x, f_x_, decimal=10)
                            npt.assert_almost_equal(f_y, f_y_, decimal=10)

    def test_hessian(self):
        for gamma in self.gamma_list:
            for e1 in self.e1_list:
                for e2 in self.e2_list:
                    for theta_E in self.theta_E_list:
                        for a4_a in self.a4_a_list:
                            kwargs_epl = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                            }
                            phi, q = param_util.ellipticity2phi_q(e1, e2)
                            kwargs_multipole = {
                                "m": 4,
                                "a_m": a4_a * theta_E / np.sqrt(q),
                                "phi_m": phi,
                            }
                            kwargs_epl_boxydisky = {
                                "theta_E": theta_E,
                                "gamma": gamma,
                                "e1": e1,
                                "e2": e2,
                                "a4_a": a4_a,
                            }
                            f_xx1, f_xy1, f_yx1, f_yy1 = self.epl.hessian(
                                self.x, self.y, **kwargs_epl
                            )
                            f_xx2, f_xy2, f_yx2, f_yy2 = self.multipole.hessian(
                                self.x, self.y, **kwargs_multipole
                            )
                            f_xx = f_xx1 + f_xx2
                            f_xy = f_xy1 + f_xy2
                            f_yx = f_yx1 + f_yx2
                            f_yy = f_yy1 + f_yy2
                            f_xx_, f_xy_, f_yx_, f_yy_ = self.epl_boxydisky.hessian(
                                self.x, self.y, **kwargs_epl_boxydisky
                            )

                            npt.assert_almost_equal(f_xx, f_xx_, decimal=10)
                            npt.assert_almost_equal(f_xy, f_xy_, decimal=10)
                            npt.assert_almost_equal(f_yx, f_yx_, decimal=10)
                            npt.assert_almost_equal(f_yy, f_yy_, decimal=10)


if __name__ == "__main__":
    pytest.main()

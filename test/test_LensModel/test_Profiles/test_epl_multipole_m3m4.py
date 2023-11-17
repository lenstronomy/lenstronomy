import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.epl import EPL
from lenstronomy.LensModel.Profiles.multipole import Multipole
from lenstronomy.LensModel.Profiles.epl_multipole_m3m4 import EPL_MULTIPOLE_M3M4
from lenstronomy.LensModel.lens_model import LensModel


class TestEPL_MULTIPOLE_M3M4(object):
    """Test TestEPL_MULTIPOLE_M3M4 vs EPL + 2 MULTIPOLE values."""

    def setup_method(self):
        self.epl = EPL()
        self.multipole = Multipole()
        self.epl_multipole = EPL_MULTIPOLE_M3M4()
        self.x, self.y = util.make_grid(numPix=10, deltapix=0.2)
        self.lens_model_split = LensModel(['EPL','MULTIPOLE','MULTIPOLE'])
        self.lens_model = LensModel(['EPL_MULTIPOLE_M3M4'])
        self.lens_model_eplboxydisky = LensModel(['EPL_BOXYDISKY'])

    def test_function(self):

        a3_a = 0.008
        a4_a = -0.01
        delta_phi_m3 = np.pi/8
        delta_phi_m4 = -np.pi / 6
        kwargs_epl = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                     'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4 = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                     'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4['a4_a'] = a4_a
        kwargs_epl_multipole_m3m4['a3_a'] = a3_a
        kwargs_epl_multipole_m3m4['delta_phi_m3'] = delta_phi_m3
        kwargs_epl_multipole_m3m4['delta_phi_m4'] = delta_phi_m4
        kwargs_epl_multipole_m3m4 = [kwargs_epl_multipole_m3m4]

        phi_q, q = param_util.ellipticity2phi_q(kwargs_epl_multipole_m3m4[0]['e1'], kwargs_epl_multipole_m3m4[0]['e2'])
        rescale = kwargs_epl_multipole_m3m4[0]['theta_E'] / np.sqrt(q)
        kwargs_multliple_m3 = {'m': 3, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a3_a * rescale,
                               'phi_m': phi_q + delta_phi_m3}
        kwargs_multliple_m4 = {'m': 4, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a4_a * rescale,
                               'phi_m': phi_q + delta_phi_m4}
        kwargs_split = [kwargs_epl, kwargs_multliple_m3, kwargs_multliple_m4]
        function_joint = self.lens_model.potential(self.x, self.y, kwargs_epl_multipole_m3m4)
        function_split = self.lens_model_split.potential(self.x, self.y, kwargs_split)
        npt.assert_allclose(function_joint, function_split)

        kwargs_epl_boxydisky = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                     'e1': 0.1, 'e2': 0.2, 'gamma': 2.0, 'a4_a': a4_a}
        kwargs_epl_multipole_m3m4[0]['a3_a'] = 0.0
        kwargs_epl_multipole_m3m4[0]['delta_phi_m4'] = 0.0
        function_joint = self.lens_model.potential(self.x, self.y, kwargs_epl_multipole_m3m4)
        function_eplboxy_disky = self.lens_model_eplboxydisky.potential(self.x, self.y, [kwargs_epl_boxydisky])
        npt.assert_allclose(function_joint, function_eplboxy_disky)

    def test_derivatives(self):
        a3_a = 0.008
        a4_a = -0.01
        delta_phi_m3 = np.pi / 8
        delta_phi_m4 = -np.pi / 6
        kwargs_epl = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                      'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4 = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                     'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4['a4_a'] = a4_a
        kwargs_epl_multipole_m3m4['a3_a'] = a3_a
        kwargs_epl_multipole_m3m4['delta_phi_m3'] = delta_phi_m3
        kwargs_epl_multipole_m3m4['delta_phi_m4'] = delta_phi_m4
        kwargs_epl_multipole_m3m4 = [kwargs_epl_multipole_m3m4]

        phi_q, q = param_util.ellipticity2phi_q(kwargs_epl_multipole_m3m4[0]['e1'], kwargs_epl_multipole_m3m4[0]['e2'])
        rescale = kwargs_epl_multipole_m3m4[0]['theta_E'] / np.sqrt(q)
        kwargs_multliple_m3 = {'m': 3, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a3_a * rescale,
                               'phi_m': phi_q + delta_phi_m3}
        kwargs_multliple_m4 = {'m': 4, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a4_a * rescale,
                               'phi_m': phi_q + delta_phi_m4}
        kwargs_split = [kwargs_epl, kwargs_multliple_m3, kwargs_multliple_m4]
        alpha_x, alpha_y = self.lens_model.alpha(self.x, self.y, kwargs_epl_multipole_m3m4)
        alpha_x_split, alpha_y_split = self.lens_model_split.alpha(self.x, self.y, kwargs_split)
        npt.assert_allclose(alpha_x, alpha_x_split)
        npt.assert_allclose(alpha_y, alpha_y_split)

        kwargs_epl_boxydisky = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                'e1': 0.1, 'e2': 0.2, 'gamma': 2.0, 'a4_a': a4_a}
        kwargs_epl_multipole_m3m4[0]['a3_a'] = 0.0
        kwargs_epl_multipole_m3m4[0]['delta_phi_m4'] = 0.0
        alpha_x_joint, alpha_y_joint = self.lens_model.alpha(self.x, self.y, kwargs_epl_multipole_m3m4)
        alpha_x_split, alpha_y_split = self.lens_model_eplboxydisky.alpha(self.x, self.y, [kwargs_epl_boxydisky])
        npt.assert_allclose(alpha_x_joint, alpha_x_split)
        npt.assert_allclose(alpha_y_joint, alpha_y_split)

    def test_hessian(self):
        a3_a = 0.008
        a4_a = -0.01
        delta_phi_m3 = np.pi / 8
        delta_phi_m4 = -np.pi / 6
        kwargs_epl = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                      'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4 = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                     'e1': 0.1, 'e2': 0.2, 'gamma': 2.0}
        kwargs_epl_multipole_m3m4['a4_a'] = a4_a
        kwargs_epl_multipole_m3m4['a3_a'] = a3_a
        kwargs_epl_multipole_m3m4['delta_phi_m3'] = delta_phi_m3
        kwargs_epl_multipole_m3m4['delta_phi_m4'] = delta_phi_m4
        kwargs_epl_multipole_m3m4 = [kwargs_epl_multipole_m3m4]

        phi_q, q = param_util.ellipticity2phi_q(kwargs_epl_multipole_m3m4[0]['e1'], kwargs_epl_multipole_m3m4[0]['e2'])
        rescale = kwargs_epl_multipole_m3m4[0]['theta_E'] / np.sqrt(q)
        kwargs_multliple_m3 = {'m': 3, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a3_a * rescale,
                               'phi_m': phi_q + delta_phi_m3}
        kwargs_multliple_m4 = {'m': 4, 'center_x': 0.1, 'center_y': 0.0,
                               'a_m': a4_a * rescale,
                               'phi_m': phi_q + delta_phi_m4}
        kwargs_split = [kwargs_epl, kwargs_multliple_m3, kwargs_multliple_m4]
        fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x, self.y, kwargs_epl_multipole_m3m4)
        fxx_split, fxy_split, fyx_split, fyy_split = self.lens_model_split.hessian(self.x, self.y, kwargs_split)
        npt.assert_allclose(fxx, fxx_split)
        npt.assert_allclose(fxy, fxy_split)
        npt.assert_allclose(fyx, fyx_split)
        npt.assert_allclose(fyy, fyy_split)

        kwargs_epl_boxydisky = {'theta_E': 1.2, 'center_x': 0.1, 'center_y': 0.0,
                                'e1': 0.1, 'e2': 0.2, 'gamma': 2.0, 'a4_a': a4_a}
        kwargs_epl_multipole_m3m4[0]['a3_a'] = 0.0
        kwargs_epl_multipole_m3m4[0]['delta_phi_m4'] = 0.0
        fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x, self.y, kwargs_epl_multipole_m3m4)
        fxx_split, fxy_split, fyx_split, fyy_split = self.lens_model_eplboxydisky.hessian(self.x, self.y, [kwargs_epl_boxydisky])
        npt.assert_allclose(fxx, fxx_split)
        npt.assert_allclose(fxy, fxy_split)
        npt.assert_allclose(fyx, fyx_split)
        npt.assert_allclose(fyy, fyy_split)


if __name__ == "__main__":
    pytest.main()

__author__ = "dangilman"
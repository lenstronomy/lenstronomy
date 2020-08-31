__author__ = 'sibirrer'


import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util


class TestEPL(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        from lenstronomy.LensModel.Profiles.epl import EPL
        self.EPL = EPL()
        from lenstronomy.LensModel.Profiles.nie import NIE
        self.NIE = NIE()

    def test_function(self):
        phi_E = 1.
        gamma = 2.
        q = 0.999
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, e1, e2, gamma)
        values_nie = self.NIE.function(x, y, phi_E, e1, e2, 0.)
        delta_f = values[0] - values[1]
        delta_f_nie = values_nie[0] - values_nie[1]
        npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)

        q = 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, e1, e2, gamma)
        values_nie = self.NIE.function(x, y, phi_E, e1, e2, 0.)
        delta_f = values[0] - values[1]
        delta_f_nie = values_nie[0] - values_nie[1]
        npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)

        q = 0.4
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, e1, e2, gamma)
        values_nie = self.NIE.function(x, y, phi_E, e1, e2, 0.)
        delta_f = values[0] - values[1]
        delta_f_nie = values_nie[0] - values_nie[1]
        npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 2.
        q = 1.
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, e1, e2, gamma)
        f_x_nie, f_y_nie = self.NIE.derivatives(x, y, phi_E, e1, e2, 0.)
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

        q = 0.7
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, e1, e2, gamma)
        f_x_nie, f_y_nie = self.NIE.derivatives(x, y, phi_E, e1, e2, 0.)
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

    def test_hessian(self):
        x = np.array([1.])
        y = np.array([2.])
        phi_E = 1.
        gamma = 2.
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_yy,f_xy = self.EPL.hessian(x, y, phi_E, e1, e2, gamma)
        f_xx_nie, f_yy_nie, f_xy_nie = self.NIE.hessian(x, y, phi_E, e1, e2, 0.)
        npt.assert_almost_equal(f_xx, f_xx_nie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_nie, decimal=4)

    def test_static(self):
        x, y = 1., 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {'theta_E': 1., 'gamma': 1.5, 'e1': e1, 'e2': e2}
        f_ = self.EPL.function(x, y, **kwargs_lens)
        self.EPL.set_static(**kwargs_lens)
        f_static = self.EPL.function(x, y, **kwargs_lens)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        self.EPL.set_dynamic()
        kwargs_lens = {'theta_E': 2., 'gamma': 1.9, 'e1': e1, 'e2': e2}
        f_dyn = self.EPL.function(x, y, **kwargs_lens)
        assert f_dyn != f_static


if __name__ == '__main__':
    pytest.main()

__author__ = 'sibirrer'


import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util

try:
    import fastell4py
    fastell4py_bool = True
except:
    print("Warning: fastell4py not available, tests will be trivially fulfilled without giving the right answer!")
    fastell4py_bool = False


class TestSPEMD(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        from lenstronomy.LensModel.Profiles.spemd_smooth import SPEMD_SMOOTH
        self.SPEMD_SMOOT = SPEMD_SMOOTH()
        from lenstronomy.LensModel.Profiles.nie import NIE
        self.NIE = NIE()

    def test_function(self):
        phi_E = 1.
        gamma = 2.
        q = 0.999
        phi_G = 1.
        s_scale = 0.1
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.SPEMD_SMOOT.function(x, y, phi_E, gamma, e1, e2, s_scale)
        if fastell4py_bool:
            values_nie = self.NIE.function(x, y, phi_E, e1, e2, s_scale)
            delta_f = values[0] - values[1]
            delta_f_nie = values_nie[0] - values_nie[1]
            npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)
        else:
            npt.assert_almost_equal(values, 0, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 2.
        q = 0.7
        phi_G = 1.
        s_scale = 0.1
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.SPEMD_SMOOT.derivatives(x, y, phi_E, gamma, e1, e2, s_scale)
        if fastell4py_bool:
            f_x_nie, f_y_nie = self.NIE.derivatives(x, y, phi_E, e1, e2, s_scale)
            #TODO test with higher precision, convention of theta_E might be slightly different from NIE with SPEMD
            npt.assert_almost_equal(f_x, f_x_nie, decimal=2)
            npt.assert_almost_equal(f_y, f_y_nie, decimal=2)
        else:
            npt.assert_almost_equal(f_x, 0, decimal=7)
            npt.assert_almost_equal(f_y, 0, decimal=7)

    def test_hessian(self):
        x = np.array([1.])
        y = np.array([2.])
        phi_E = 1.
        gamma = 2.
        q = 0.9
        phi_G = 1.
        s_scale = 0.1
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_yy,f_xy = self.SPEMD_SMOOT.hessian(x, y, phi_E, gamma, e1, e2, s_scale)
        if fastell4py_bool:
            f_xx_nie, f_yy_nie, f_xy_nie = self.NIE.hessian(x, y, phi_E, e1, e2, s_scale)
            npt.assert_almost_equal(f_xx, f_xx_nie, decimal=3)
            npt.assert_almost_equal(f_yy, f_yy_nie, decimal=3)
            npt.assert_almost_equal(f_xy, f_xy_nie, decimal=3)
        else:
            npt.assert_almost_equal(f_xx, 0, decimal=7)
            npt.assert_almost_equal(f_yy, 0, decimal=7)
            npt.assert_almost_equal(f_xy, 0, decimal=7)

    def test_bounds(self):
        compute_bool = self.SPEMD_SMOOT._parameter_constraints(q_fastell=-1, gam=-1, s2=-1, q=-1)
        assert compute_bool is False

    def test_is_not_empty(self):
        func = self.SPEMD_SMOOT.is_not_empty

        assert func(0.1, 0.2)
        assert func([0.1], [0.2])
        assert func((0.1, 0.3), (0.2, 0.4))
        assert func(np.array([0.1]), np.array([0.2]))
        assert not func([], [])
        assert not func(np.array([]), np.array([]))


if __name__ == '__main__':
    pytest.main()

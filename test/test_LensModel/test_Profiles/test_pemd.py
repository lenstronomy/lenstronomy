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
    def setup_method(self):
        from lenstronomy.LensModel.Profiles.pemd import PEMD
        from lenstronomy.LensModel.Profiles.spep import SPEP
        self.PEMD = PEMD(suppress_fastell=True)
        self.SPEP = SPEP()

    def test_function(self):
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1.])
        y = np.array([2])
        a = np.zeros_like(x)
        values = self.PEMD.function(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(values[0], 2.1571106351401803, decimal=5)
        else:
            assert values == 0
        a += values
        x = np.array(1.)
        y = np.array(2.)
        a = np.zeros_like(x)
        values = self.PEMD.function(x, y, phi_E, gamma, e1, e2)
        print(x, values)
        a += values
        if fastell4py_bool:
            npt.assert_almost_equal(values, 2.1571106351401803, decimal=5)
        else:
            assert values == 0
        assert type(x) == type(values)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.PEMD.function(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(values[0], 2.180188584782964, decimal=7)
            npt.assert_almost_equal(values[1], 3.2097137160951874, decimal=7)
            npt.assert_almost_equal(values[2], 4.3109976673748, decimal=7)
        else:
            npt.assert_almost_equal(values[0], 0, decimal=7)
            npt.assert_almost_equal(values[1], 0, decimal=7)
            npt.assert_almost_equal(values[2], 0, decimal=7)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.PEMD.derivatives(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_x[0], 0.46664118422711387, decimal=7)
            npt.assert_almost_equal(f_y[0], 0.9530892465981603, decimal=7)
        else:
            npt.assert_almost_equal(f_x[0], 0, decimal=7)
            npt.assert_almost_equal(f_y[0], 0, decimal=7)

        x = np.array([1., 3, 4])
        y = np.array([2., 1, 1])
        a = np.zeros_like(x)
        values = self.PEMD.derivatives(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(values[0][0], 0.46664118422711387, decimal=7)
            npt.assert_almost_equal(values[1][0], 0.9530892465981603, decimal=7)
            npt.assert_almost_equal(values[0][1], 1.0722265330847958, decimal=7)
            npt.assert_almost_equal(values[1][1], 0.3140067377020791, decimal=7)
        else:
            npt.assert_almost_equal(values[0][0], 0, decimal=7)
            npt.assert_almost_equal(values[1][0], 0, decimal=7)
            npt.assert_almost_equal(values[0][1], 0, decimal=7)
            npt.assert_almost_equal(values[1][1], 0, decimal=7)
        a += values[0]
        x = 1.
        y = 2.
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.PEMD.derivatives(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_x, 0.46664118422711387, decimal=7)
            npt.assert_almost_equal(f_y, 0.9530892465981603, decimal=7)
        else:
            npt.assert_almost_equal(f_x, 0, decimal=7)
            npt.assert_almost_equal(f_y, 0, decimal=7)
        x = 0.
        y = 0.
        f_x, f_y = self.PEMD.derivatives(x, y, phi_E, gamma, e1, e2)
        assert f_x == 0.
        assert f_y == 0.

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.PEMD.hessian(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_xx, 0.4179041, decimal=7)
            npt.assert_almost_equal(f_yy, 0.1404714, decimal=7)
            npt.assert_almost_equal(f_xy, -0.1856134, decimal=7)
        else:
            npt.assert_almost_equal(f_xx, 0, decimal=7)
            npt.assert_almost_equal(f_yy, 0, decimal=7)
            npt.assert_almost_equal(f_xy, 0, decimal=7)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

        x = 1.
        y = 2.
        phi_E = 1.
        gamma = 1.9
        q = 0.9
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        a = np.zeros_like(x)
        f_xx, f_xy, f_yx, f_yy = self.PEMD.hessian(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_xx, 0.41790408341142493, decimal=7)
            npt.assert_almost_equal(f_yy, 0.14047143086334482, decimal=7)
            npt.assert_almost_equal(f_xy, -0.1856133848300859, decimal=7)
        else:
            npt.assert_almost_equal(f_xx, 0, decimal=7)
            npt.assert_almost_equal(f_yy, 0, decimal=7)
            npt.assert_almost_equal(f_xy, 0, decimal=7)
        a += f_xx
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.PEMD.hessian(x, y, phi_E, gamma, e1, e2)
        print(values, 'values')
        if fastell4py_bool:
            npt.assert_almost_equal(values[0][0], 0.41789957732890953, decimal=5)
            npt.assert_almost_equal(values[3][0], 0.14047593655054141, decimal=5)
            npt.assert_almost_equal(values[1][0], -0.18560737698052343, decimal=5)
            npt.assert_almost_equal(values[0][1], 0.068359818958208918, decimal=5)
            npt.assert_almost_equal(values[3][1], 0.32494089371516482, decimal=5)
            npt.assert_almost_equal(values[1][1], -0.097845438684594374, decimal=5)
        else:
            npt.assert_almost_equal(values[0][0], 0, decimal=7)

    def test_spep_spemd(self):
        x = np.array([1])
        y = np.array([0])
        phi_E = 1.
        gamma = 2.
        q = 1.
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.PEMD.derivatives(x, y, phi_E, gamma, e1, e2)
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, phi_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_x[0], f_x_spep[0], decimal=2)
        else:
            pass

        theta_E = 2.
        gamma = 2.
        q = 1.
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.PEMD.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, theta_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_x[0], f_x_spep[0], decimal=2)
        else:
            pass

        theta_E = 2.
        gamma = 1.7
        q = 1.
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.PEMD.derivatives(x, y, theta_E, gamma, e1, e2)
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, theta_E, gamma, e1, e2)
        if fastell4py_bool:
            npt.assert_almost_equal(f_x[0], f_x_spep[0], decimal=4)

    def test_bounds(self):
        from lenstronomy.LensModel.Profiles.spemd import SPEMD
        profile = SPEMD(suppress_fastell=True)
        compute_bool = profile._parameter_constraints(q_fastell=-1, gam=-1, s2=-1, q=-1)
        assert compute_bool is False

    def test_is_not_empty(self):
        func = self.PEMD.spemd_smooth.is_not_empty

        assert func(0.1, 0.2)
        assert func([0.1], [0.2])
        assert func((0.1, 0.3), (0.2, 0.4))
        assert func(np.array([0.1]), np.array([0.2]))
        assert not func([], [])
        assert not func(np.array([]), np.array([]))

    def test_density_lens(self):
        r = 1
        kwargs = {'theta_E': 1, 'gamma': 2, 'e1': 0, 'e2': 0}
        rho = self.PEMD.density_lens(r, **kwargs)
        rho_spep = self.SPEP.density_lens(r, **kwargs)
        npt.assert_almost_equal(rho, rho_spep, decimal=7)


if __name__ == '__main__':
    pytest.main()


class TestPEMD_QPHI(object):
    """
    tests the q phi methods
    """

    def setup_method(self):
        from lenstronomy.LensModel.Profiles.pemd import PEMD_qPhi
        from lenstronomy.LensModel.Profiles.pemd import PEMD
        self.PEMD = PEMD()
        self.PEMD_qPhi = PEMD_qPhi()

    def test_function(self):
        theta_E = 1.
        gamma = 1.9
        q = 0.9
        phi = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        x = np.array([1.])
        y = np.array([2])
        val_PEMD=self.PEMD.function(x, y, theta_E, gamma, e1, e2)
        val_PEMD_Q_PHI=self.PEMD_qPhi.function(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(val_PEMD, val_PEMD_Q_PHI, decimal=7)

    def test_derivatives(self):
        theta_E = 1.
        gamma = 1.9
        q = 0.9
        phi = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        x = np.array([1.])
        y = np.array([2])
        val_PEMD=self.PEMD.derivatives(x, y, theta_E, gamma, e1, e2)
        val_PEMD_Q_PHI=self.PEMD_qPhi.derivatives(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(val_PEMD, val_PEMD_Q_PHI, decimal=7)

    def test_hessian(self):
        theta_E = 1.
        gamma = 1.9
        q = 0.9
        phi = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        x = np.array([1.])
        y = np.array([2])
        val_PEMD=self.PEMD.hessian(x, y, theta_E, gamma, e1, e2)
        val_PEMD_Q_PHI=self.PEMD_qPhi.hessian(x, y, theta_E, gamma, q, phi)
        npt.assert_almost_equal(val_PEMD, val_PEMD_Q_PHI, decimal=7)
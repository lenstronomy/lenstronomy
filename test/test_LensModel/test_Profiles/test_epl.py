__author__ = 'sibirrer'


import numpy as np
import pytest
import numpy.testing as npt
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util

try:
    import fastell4py
    fastell4py_bool = True
except:
    print("Warning: fastell4py not available, tests will be trivially fulfilled without giving the right answer!")
    fastell4py_bool = False


class TestEPLvsNIE(object):
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
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
        values_nie = self.NIE.function(x, y, phi_E, e1, e2, 0.)
        delta_f = values[0] - values[1]
        delta_f_nie = values_nie[0] - values_nie[1]
        npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)

        q = 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
        values_nie = self.NIE.function(x, y, phi_E, e1, e2, 0.)
        delta_f = values[0] - values[1]
        delta_f_nie = values_nie[0] - values_nie[1]
        npt.assert_almost_equal(delta_f, delta_f_nie, decimal=5)

        q = 0.4
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        x = np.array([1., 2])
        y = np.array([2, 0])
        values = self.EPL.function(x, y, phi_E, gamma, e1, e2)
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
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
        f_x_nie, f_y_nie = self.NIE.derivatives(x, y, phi_E, e1, e2, 0.)
        npt.assert_almost_equal(f_x, f_x_nie, decimal=4)
        npt.assert_almost_equal(f_y, f_y_nie, decimal=4)

        q = 0.7
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
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
        f_xx, f_xy, f_yx, f_yy = self.EPL.hessian(x, y, phi_E, gamma, e1, e2)
        f_xx_nie, f_xy_nie, f_yx_nie, f_yy_nie = self.NIE.hessian(x, y, phi_E, e1, e2, 0.)
        npt.assert_almost_equal(f_xx, f_xx_nie, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_nie, decimal=4)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

    def test_density_lens(self):
        r = 1
        kwargs = {'theta_E': 1, 'gamma': 2, 'e1': 0, 'e2': 0}
        rho = self.EPL.density_lens(r, **kwargs)
        from lenstronomy.LensModel.Profiles.spep import SPEP
        spep = SPEP()
        rho_spep = spep.density_lens(r, **kwargs)
        npt.assert_almost_equal(rho, rho_spep, decimal=7)

    def test_mass_3d_lens(self):
        r = 1
        kwargs = {'theta_E': 1, 'gamma': 2, 'e1': 0, 'e2': 0}
        rho = self.EPL.mass_3d_lens(r, **kwargs)
        from lenstronomy.LensModel.Profiles.spep import SPEP
        spep = SPEP()
        rho_spep = spep.mass_3d_lens(r, **kwargs)
        npt.assert_almost_equal(rho, rho_spep, decimal=7)

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

    def test_regularization(self):

        phi_E = 1.
        gamma = 2.
        q = 1.
        phi_G = 1.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        x = 0.
        y = 0.
        f_x, f_y = self.EPL.derivatives(x, y, phi_E, gamma, e1, e2)
        npt.assert_almost_equal(f_x, 0.)
        npt.assert_almost_equal(f_y, 0.)

        x = 0.
        y = 0.
        f_xx, f_xy, f_yx, f_yy = self.EPL.hessian(x, y, phi_E, gamma, e1, e2)
        assert f_xx > 10 ** 5
        assert f_yy > 10 ** 5
        #npt.assert_almost_equal(f_xx, 10**10)
        #npt.assert_almost_equal(f_yy, 10**10)
        npt.assert_almost_equal(f_xy, 0)
        npt.assert_almost_equal(f_yx, 0)


class TestEPLvsPEMD(object):
    """
    Test EPL model vs PEMD with FASTELL
    This tests get only executed if fastell is installed
    """
    def setup(self):
        try:
            import fastell4py
            self._fastell4py_bool = True
        except:
            print("Warning: fastell4py not available, tests will be trivially fulfilled without giving the right answer!")
            self._fastell4py_bool = False
        from lenstronomy.LensModel.Profiles.epl import EPL
        self.epl = EPL()
        from lenstronomy.LensModel.Profiles.pemd import PEMD
        self.pemd = PEMD(suppress_fastell=True)

    def test_epl_pemd_convention(self):
        """
        tests convention of EPL and PEMD model on the deflection angle basis
        """
        if self._fastell4py_bool is False:
            return 0
        x, y = util.make_grid(numPix=10, deltapix=0.2)

        theta_E_list = [0.5, 1, 2]
        gamma_list = [1.8, 2., 2.2]
        e1_list = [-0.2, 0., 0.2]
        e2_list = [-0.2, 0., 0.2]
        for gamma in gamma_list:
            for e1 in e1_list:
                for e2 in e2_list:
                    for theta_E in theta_E_list:
                        kwargs = {'theta_E': theta_E, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}
                        f_x, f_y = self.epl.derivatives(x, y, **kwargs)
                        f_x_pemd, f_y_pemd = self.pemd.derivatives(x, y, **kwargs)

                        npt.assert_almost_equal(f_x, f_x_pemd, decimal=4)
                        npt.assert_almost_equal(f_y, f_y_pemd, decimal=4)


if __name__ == '__main__':
    pytest.main()

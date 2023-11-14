__author__ = "sibirrer"


import numpy as np
import numpy.testing as npt
import pytest
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util
from lenstronomy.LensModel.Profiles.nie import NIE, NIEMajorAxis
from lenstronomy.LensModel.Profiles.spemd import SPEMD
from lenstronomy.LensModel.Profiles.sis import SIS

try:
    import fastell4py

    bool_test = True
except:
    bool_test = False
    print(
        "Warning: fastell4py not available, tests will not crosscheck with fastell4py on your machine"
    )


class TestNIE(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        self.nie = NIE()
        self.spemd = SPEMD(suppress_fastell=True)
        self.sis = SIS()

    def test_function(self):
        y = np.array([1.0, 2])
        x = np.array([0.0, 0.0])
        theta_E = 1.0
        q = 0.9999
        s = 0.00001
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)

        values = self.nie.function(x, y, theta_E, e1, e2, s_scale=s)
        delta_pot = values[1] - values[0]
        values_spemd = self.sis.function(x, y, theta_E)
        delta_pot_spemd = values_spemd[1] - values_spemd[0]
        npt.assert_almost_equal(delta_pot, delta_pot_spemd, decimal=4)
        if bool_test is True:
            q = 0.99
            s = 0.000001
            phi_G = 0
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            values = self.nie.function(x, y, theta_E, e1, e2, s_scale=s)
            delta_pot = values[1] - values[0]
            gamma = 2.0
            values_spemd = self.spemd.function(x, y, theta_E, gamma, e1, e2, s_scale=s)
            delta_pot_spemd = values_spemd[1] - values_spemd[0]
            npt.assert_almost_equal(delta_pot, delta_pot_spemd, decimal=2)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        q = 0.99999
        phi_G = 0
        s = 0.0000001
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale=s)
        f_x_spemd, f_y_spemd = self.sis.derivatives(x, y, theta_E)
        npt.assert_almost_equal(f_x, f_x_spemd, decimal=4)
        npt.assert_almost_equal(f_y, f_y_spemd, decimal=4)
        if bool_test is True:
            q = 0.99
            s = 0.000001
            phi_G = 0
            e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
            f_x, f_y = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale=s)
            gamma = 2.0
            f_x_spemd, f_y_spemd = self.spemd.derivatives(
                x, y, theta_E, gamma, e1, e2, s_scale=s
            )
            print(f_x / f_x_spemd, "ratio deflections")
            print(1 + (1 - q) / 2)
            npt.assert_almost_equal(f_x, f_x_spemd, decimal=2)
            npt.assert_almost_equal(f_y, f_y_spemd, decimal=2)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.0
        q = 0.999999
        phi_G = 0
        s = 0.0000001
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_xy, f_yx, f_yy = self.nie.hessian(x, y, theta_E, e1, e2, s_scale=s)
        f_xx_spemd, f_xy_spemd, f_yx_spemd, f_yy_spemd = self.sis.hessian(x, y, theta_E)
        npt.assert_almost_equal(f_xx, f_xx_spemd, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_spemd, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_spemd, decimal=4)
        npt.assert_almost_equal(f_yx, f_yx_spemd, decimal=4)

    def test_convergence2surface_brightness(self):
        from lenstronomy.LightModel.Profiles.nie import NIE as NIE_Light

        nie_light = NIE_Light()
        kwargs = {"e1": 0.3, "e2": -0.05, "s_scale": 0.5}
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        f_xx, f_xy, f_yx, f_yy = self.nie.hessian(x, y, theta_E=1, **kwargs)
        kappa = 1 / 2.0 * (f_xx + f_yy)
        flux = nie_light.function(x, y, amp=1, **kwargs)
        npt.assert_almost_equal(kappa / np.sum(kappa), flux / np.sum(flux), decimal=5)

    def test_static(self):
        x, y = 1.0, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {"theta_E": 1.0, "s_scale": 0.1, "e1": e1, "e2": e2}
        f_ = self.nie.function(x, y, **kwargs_lens)
        self.nie.set_static(**kwargs_lens)
        f_static = self.nie.function(x, y, **kwargs_lens)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        self.nie.set_dynamic()
        kwargs_lens = {"theta_E": 2.0, "s_scale": 0.1, "e1": e1, "e2": e2}
        f_dyn = self.nie.function(x, y, **kwargs_lens)
        assert f_dyn != f_static


class TestNIEMajorAxis(object):
    def setup_method(self):
        pass

    def test_kappa(self):
        nie = NIEMajorAxis()
        x, y = util.make_grid(numPix=10, deltapix=0.1)
        kwargs = {"b": 1, "s": 0.2, "q": 0.3}
        f_xx, f_xy, f_yx, f_yy = nie.hessian(x, y, **kwargs)
        kappa_num = 1.0 / 2 * (f_xx + f_yy)
        kappa = nie.kappa(x, y, **kwargs)
        npt.assert_almost_equal(kappa_num, kappa, decimal=5)


if __name__ == "__main__":
    pytest.main()

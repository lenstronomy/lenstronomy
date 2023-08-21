import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.chameleon import (
    Chameleon,
    DoubleChameleon,
    DoubleChameleonPointMass,
    TripleChameleon,
)
from lenstronomy.LightModel.Profiles.chameleon import (
    DoubleChameleon as DoubleChameleonLight,
)
from lenstronomy.LightModel.Profiles.chameleon import (
    TripleChameleon as TripleChameleonLight,
)
import lenstronomy.Util.param_util as param_util


class TestChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        self.chameleon = Chameleon()
        self.nie = NIE()

    def test_theta_E_convert(self):
        w_c, w_t = 2, 1
        theta_E_convert, w_c, w_t, s_scale_1, s_scale_2 = self.chameleon.param_convert(
            alpha_1=1, w_c=w_c, w_t=w_t, e1=0, e2=0
        )
        assert w_c == 1
        assert w_t == 2
        assert theta_E_convert == 0

    def test_function(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {"alpha_1": 1.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        theta_E_convert, w_c, w_t, s_scale_1, s_scale_2 = self.chameleon.param_convert(
            alpha_1=1, w_c=0.5, w_t=1.0, e1=e1, e2=e2
        )
        kwargs_1 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_1,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_2,
            "e1": e1,
            "e2": e2,
        }
        f_ = self.chameleon.function(x=x, y=1.0, **kwargs_light)
        f_1 = self.nie.function(x=x, y=1.0, **kwargs_1)
        f_2 = self.nie.function(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(f_, (f_1 - f_2), decimal=5)

    def test_derivatives(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {"alpha_1": 1.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        theta_E_convert, w_c, w_t, s_scale_1, s_scale_2 = self.chameleon.param_convert(
            alpha_1=1, w_c=0.5, w_t=1.0, e1=e1, e2=e2
        )
        kwargs_1 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_1,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_2,
            "e1": e1,
            "e2": e2,
        }
        f_x, f_y = self.chameleon.derivatives(x=x, y=1.0, **kwargs_light)
        f_x_1, f_y_1 = self.nie.derivatives(x=x, y=1.0, **kwargs_1)
        f_x_2, f_y_2 = self.nie.derivatives(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(f_x, (f_x_1 - f_x_2), decimal=5)
        npt.assert_almost_equal(f_y, (f_y_1 - f_y_2), decimal=5)
        f_x, f_y = self.chameleon.derivatives(x=1, y=0.0, **kwargs_light)
        npt.assert_almost_equal(f_x, 1, decimal=1)

    def test_hessian(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {"alpha_1": 1.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        theta_E_convert, w_c, w_t, s_scale_1, s_scale_2 = self.chameleon.param_convert(
            alpha_1=1, w_c=0.5, w_t=1.0, e1=e1, e2=e2
        )
        kwargs_1 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_1,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "theta_E": theta_E_convert,
            "s_scale": s_scale_2,
            "e1": e1,
            "e2": e2,
        }
        f_xx, f_xy, f_yx, f_yy = self.chameleon.hessian(x=x, y=1.0, **kwargs_light)
        f_xx_1, f_xy_1, f_yx_1, f_yy_1 = self.nie.hessian(x=x, y=1.0, **kwargs_1)
        f_xx_2, f_xy_2, f_yx_2, f_yy_2 = self.nie.hessian(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(f_xx, (f_xx_1 - f_xx_2), decimal=5)
        npt.assert_almost_equal(f_yy, (f_yy_1 - f_yy_2), decimal=5)
        npt.assert_almost_equal(f_xy, (f_xy_1 - f_xy_2), decimal=5)
        npt.assert_almost_equal(f_yx, (f_yx_1 - f_yx_2), decimal=5)

    def test_static(self):
        x, y = 1.0, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {"alpha_1": 1.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        f_ = self.chameleon.function(x, y, **kwargs_light)
        self.chameleon.set_static(**kwargs_light)
        f_static = self.chameleon.function(x, y, **kwargs_light)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        self.chameleon.set_dynamic()
        kwargs_light = {"alpha_1": 2.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        f_dyn = self.chameleon.function(x, y, **kwargs_light)
        assert f_dyn != f_static


class TestDoubleChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == "alpha_1"

    def test_function(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        theta_E = 1.0
        ratio = 2.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio": 2,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }

        kwargs_1 = {
            "alpha_1": theta_E / (1 + 1.0 / ratio),
            "w_c": 0.5,
            "w_t": 1.0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "alpha_1": theta_E / (1 + ratio),
            "w_c": 0.1,
            "w_t": 0.5,
            "e1": e1,
            "e2": e2,
        }
        flux = doublechameleon.function(x=x, y=1.0, **kwargs_light)
        flux1 = chameleon.function(x=x, y=1.0, **kwargs_1)
        flux2 = chameleon.function(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(flux, flux1 + flux2, decimal=8)

    def test_derivatives(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        theta_E = 1.0
        ratio = 2.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio": 2,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }

        kwargs_1 = {
            "alpha_1": theta_E / (1 + 1.0 / ratio),
            "w_c": 0.5,
            "w_t": 1.0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "alpha_1": theta_E / (1 + ratio),
            "w_c": 0.1,
            "w_t": 0.5,
            "e1": e1,
            "e2": e2,
        }
        f_x, f_y = doublechameleon.derivatives(x=x, y=1.0, **kwargs_light)
        f_x1, f_y1 = chameleon.derivatives(x=x, y=1.0, **kwargs_1)
        f_x2, f_y2 = chameleon.derivatives(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(f_x, f_x1 + f_x2, decimal=8)
        npt.assert_almost_equal(f_y, f_y1 + f_y2, decimal=8)

    def test_hessian(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        theta_E = 1.0
        ratio = 2.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {
            "alpha_1": theta_E,
            "ratio": ratio,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        kwargs_light = {
            "amp": theta_E,
            "ratio": ratio,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }

        kwargs_1 = {
            "alpha_1": theta_E / (1 + 1.0 / ratio),
            "w_c": 0.5,
            "w_t": 1.0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "alpha_1": theta_E / (1 + ratio),
            "w_c": 0.1,
            "w_t": 0.5,
            "e1": e1,
            "e2": e2,
        }
        f_xx, f_xy, f_yx, f_yy = doublechameleon.hessian(x=x, y=1.0, **kwargs_lens)
        f_xx1, f_xy1, f_yx1, f_yy1 = chameleon.hessian(x=x, y=1.0, **kwargs_1)
        f_xx2, f_xy2, f_yx2, f_yy2 = chameleon.hessian(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(f_xx, f_xx1 + f_xx2, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy1 + f_yy2, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy1 + f_xy2, decimal=8)
        npt.assert_almost_equal(f_yx, f_yx1 + f_yx2, decimal=8)
        light = DoubleChameleonLight()
        f_xx, f_xy, f_yx, f_yy = doublechameleon.hessian(
            x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_lens
        )
        kappa = 1.0 / 2 * (f_xx + f_yy)
        kappa_norm = kappa / np.mean(kappa)
        flux = light.function(x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_light)
        flux_norm = flux / np.mean(flux)
        npt.assert_almost_equal(kappa_norm, flux_norm, decimal=5)

    def test_static(self):
        doublechameleon = DoubleChameleon()
        x, y = 1.0, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1,
            "ratio": 0.5,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        f_ = doublechameleon.function(x, y, **kwargs_light)
        doublechameleon.set_static(**kwargs_light)
        f_static = doublechameleon.function(x, y, **kwargs_light)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        doublechameleon.set_dynamic()
        kwargs_light = {
            "alpha_1": 2,
            "ratio": 0.5,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        f_dyn = doublechameleon.function(x, y, **kwargs_light)
        assert f_dyn != f_static


class TestDoubleChameleonPointMass(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleonPointMass()
        names = chameleon.param_names
        assert names[0] == "alpha_1"

    def test_function(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleonPointMass()

        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio_pointmass": 3,
            "ratio_chameleon": 2,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        flux = doublechameleon.function(x=1, y=1.0, **kwargs_light)
        npt.assert_almost_equal(flux, 1.2602247653486218, decimal=4)

    def test_derivatives(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleonPointMass()

        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio_pointmass": 3,
            "ratio_chameleon": 2,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        f_x, f_y = doublechameleon.derivatives(x=1, y=1.0, **kwargs_light)
        npt.assert_almost_equal(f_x, 0.43419725313692664, decimal=4)
        npt.assert_almost_equal(f_y, 0.4521464786719726, decimal=4)

    def test_hessian(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleonPointMass()

        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio_pointmass": 3,
            "ratio_chameleon": 2,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
        }
        f_xx, f_xy, f_yx, f_yy = doublechameleon.hessian(x=1, y=1.0, **kwargs_light)
        npt.assert_almost_equal(f_xx, 0.06255816336369684, decimal=4)
        npt.assert_almost_equal(f_xy, -0.3986532840628945, decimal=4)
        npt.assert_almost_equal(f_yx, -0.3986532840628945, decimal=4)
        npt.assert_almost_equal(f_yy, 0.04715726782095693, decimal=4)


class TestTripleChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = TripleChameleon()
        names = chameleon.param_names
        assert names[0] == "alpha_1"

    def test_function(self):
        """

        :return:
        """
        triplechameleon = TripleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio12 = 2.0
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }

        amp1 = 1.0 / (1.0 + 1.0 / ratio12 + 1.0 / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        kwargs_1 = {"alpha_1": amp1, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        kwargs_2 = {"alpha_1": amp2, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        kwargs_3 = {"alpha_1": amp3, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        flux = triplechameleon.function(x=x, y=1.0, **kwargs_light)
        flux1 = chameleon.function(x=x, y=1.0, **kwargs_1)
        flux2 = chameleon.function(x=x, y=1.0, **kwargs_2)
        flux3 = chameleon.function(x=x, y=1.0, **kwargs_3)
        npt.assert_almost_equal(flux, flux1 + flux2 + flux3, decimal=8)

    def test_derivatives(self):
        """

        :return:
        """
        triplechameleon = TripleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio12 = 2.0
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "alpha_1": 1.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }

        amp1 = 1.0 / (1.0 + 1.0 / ratio12 + 1.0 / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        kwargs_1 = {"alpha_1": amp1, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        kwargs_2 = {"alpha_1": amp2, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        kwargs_3 = {"alpha_1": amp3, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        f_x, f_y = triplechameleon.derivatives(x=x, y=1.0, **kwargs_light)
        f_x1, f_y1 = chameleon.derivatives(x=x, y=1.0, **kwargs_1)
        f_x2, f_y2 = chameleon.derivatives(x=x, y=1.0, **kwargs_2)
        f_x3, f_y3 = chameleon.derivatives(x=x, y=1.0, **kwargs_3)
        npt.assert_almost_equal(f_x, f_x1 + f_x2 + f_x3, decimal=8)
        npt.assert_almost_equal(f_y, f_y1 + f_y2 + f_y3, decimal=8)

    def test_hessian(self):
        """

        :return:
        """
        triplechameleon = TripleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio12 = 2.0
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {
            "alpha_1": 1.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }

        kwargs_light = {
            "amp": 1.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }

        amp1 = 1.0 / (1.0 + 1.0 / ratio12 + 1.0 / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        kwargs_1 = {"alpha_1": amp1, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        kwargs_2 = {"alpha_1": amp2, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        kwargs_3 = {"alpha_1": amp3, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}

        f_xx, f_xy, f_yx, f_yy = triplechameleon.hessian(x=x, y=1.0, **kwargs_lens)
        f_xx1, f_xy1, f_yx1, f_yy1 = chameleon.hessian(x=x, y=1.0, **kwargs_1)
        f_xx2, f_xy2, f_yx2, f_yy2 = chameleon.hessian(x=x, y=1.0, **kwargs_2)
        f_xx3, f_xy3, f_yx3, f_yy3 = chameleon.hessian(x=x, y=1.0, **kwargs_3)
        npt.assert_almost_equal(f_xx, f_xx1 + f_xx2 + f_xx3, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy1 + f_yy2 + f_yy3, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy1 + f_xy2 + f_xy3, decimal=8)
        npt.assert_almost_equal(f_yx, f_yx1 + f_yx2 + f_yx3, decimal=8)
        light = TripleChameleonLight()
        f_xx, f_xy, f_yx, f_yy = triplechameleon.hessian(
            x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_lens
        )
        kappa = 1.0 / 2 * (f_xx + f_yy)
        kappa_norm = kappa / np.mean(kappa)
        flux = light.function(x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_light)
        flux_norm = flux / np.mean(flux)
        npt.assert_almost_equal(kappa_norm, flux_norm, decimal=5)

    def test_static(self):
        triplechameleon = TripleChameleon()
        x, y = 1.0, 1.0
        phi_G, q = 0.3, 0.8
        ratio12 = 2.0
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {
            "alpha_1": 1.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }
        f_ = triplechameleon.function(x, y, **kwargs_lens)
        triplechameleon.set_static(**kwargs_lens)
        f_static = triplechameleon.function(x, y, **kwargs_lens)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        triplechameleon.set_dynamic()
        kwargs_lens = {
            "alpha_1": 2.0,
            "ratio12": ratio12,
            "ratio13": ratio13,
            "w_c1": 0.5,
            "w_t1": 1.0,
            "e11": e1,
            "e21": e2,
            "w_c2": 0.1,
            "w_t2": 0.5,
            "e12": e1,
            "e22": e2,
            "w_c3": 0.1,
            "w_t3": 0.5,
            "e13": e1,
            "e23": e2,
        }
        f_dyn = triplechameleon.function(x, y, **kwargs_lens)
        assert f_dyn != f_static


if __name__ == "__main__":
    pytest.main()

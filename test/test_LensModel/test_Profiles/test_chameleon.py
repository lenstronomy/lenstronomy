
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.chameleon import Chameleon, DoubleChameleon
from lenstronomy.LightModel.Profiles.chameleon import DoubleChameleon as DoubleChameleonLight
import lenstronomy.Util.param_util as param_util


class TestChameleon(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        self.chameleon = Chameleon()
        self.nie = NIE()

    def test_function(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        theta_E_convert = self.chameleon._theta_E_convert(theta_E=1, w_c=0.5, w_t=1.)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        kwargs_1 = {'theta_E': theta_E_convert, 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E_convert, 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        f_ = self.chameleon.function(x=x, y=1., **kwargs_light)
        f_1 = self.nie.function(x=x, y=1., **kwargs_1)
        f_2 = self.nie.function(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_, (f_1 - f_2), decimal=5)

    def test_derivatives(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        theta_E_convert = self.chameleon._theta_E_convert(theta_E=1, w_c=0.5, w_t=1.)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        kwargs_1 = {'theta_E': theta_E_convert, 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E_convert, 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        f_x, f_y = self.chameleon.derivatives(x=x, y=1., **kwargs_light)
        f_x_1, f_y_1 = self.nie.derivatives(x=x, y=1., **kwargs_1)
        f_x_2, f_y_2 = self.nie.derivatives(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_x, (f_x_1 - f_x_2), decimal=5)
        npt.assert_almost_equal(f_y, (f_y_1 - f_y_2), decimal=5)
        f_x, f_y = self.chameleon.derivatives(x=1, y=0., **kwargs_light)
        npt.assert_almost_equal(f_x, 1, decimal=1)

    def test_hessian(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        theta_E_convert = self.chameleon._theta_E_convert(theta_E=1, w_c=0.5, w_t=1.)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        kwargs_1 = {'theta_E': theta_E_convert, 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E_convert, 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        f_xx, f_yy, f_xy = self.chameleon.hessian(x=x, y=1., **kwargs_light)
        f_xx_1, f_yy_1, f_xy_1 = self.nie.hessian(x=x, y=1., **kwargs_1)
        f_xx_2, f_yy_2, f_xy_2 = self.nie.hessian(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_xx, (f_xx_1 - f_xx_2), decimal=5)
        npt.assert_almost_equal(f_yy, (f_yy_1 - f_yy_2), decimal=5)
        npt.assert_almost_equal(f_xy, (f_xy_1 - f_xy_2), decimal=5)


class TestDoubleChameleon(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == 'theta_E'

    def test_function(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        theta_E = 1.
        ratio = 2.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'ratio': 2, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2, 'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2}

        kwargs_1 = {'theta_E': theta_E / (1 + 1. / ratio), 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E / (1 + ratio), 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        flux = doublechameleon.function(x=x, y=1., **kwargs_light)
        flux1 = chameleon.function(x=x, y=1., **kwargs_1)
        flux2 = chameleon.function(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(flux, flux1 + flux2, decimal=8)

    def test_derivatives(self):
        """

        :return:
        """
        doublechameleon = DoubleChameleon()
        chameleon = Chameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        theta_E = 1.
        ratio = 2.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'ratio': 2, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2, 'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2}

        kwargs_1 = {'theta_E': theta_E / (1 + 1. / ratio), 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E / (1 + ratio), 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        f_x, f_y = doublechameleon.derivatives(x=x, y=1., **kwargs_light)
        f_x1, f_y1 = chameleon.derivatives(x=x, y=1., **kwargs_1)
        f_x2, f_y2 = chameleon.derivatives(x=x, y=1., **kwargs_2)
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
        theta_E = 1.
        ratio = 2.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = {'theta_E': theta_E, 'ratio': ratio, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2, 'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2}
        kwargs_light = {'amp': theta_E, 'ratio': ratio, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2, 'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2}

        kwargs_1 = {'theta_E': theta_E / (1 + 1./ratio), 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': theta_E / (1 + ratio), 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        f_xx, f_yy, f_xy = doublechameleon.hessian(x=x, y=1., **kwargs_lens)
        f_xx1, f_yy1, f_xy1 = chameleon.hessian(x=x, y=1., **kwargs_1)
        f_xx2, f_yy2, f_xy2 = chameleon.hessian(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_xx, f_xx1 + f_xx2, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy1 + f_yy2, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy1 + f_xy2, decimal=8)
        light = DoubleChameleonLight()
        f_xx, f_yy, f_xy = doublechameleon.hessian(x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_lens)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_norm = kappa / np.mean(kappa)
        flux = light.function(x=np.linspace(0, 1, 10), y=np.zeros(10), **kwargs_light)
        flux_norm = flux / np.mean(flux)
        npt.assert_almost_equal(kappa_norm, flux_norm, decimal=5)


if __name__ == '__main__':
    pytest.main()

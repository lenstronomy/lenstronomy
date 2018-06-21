
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.chameleon import Chameleon
import lenstronomy.Util.param_util as param_util


class TestPowerLaw(object):
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
        s_scale_1 = 4 * w_c ** 2 / (1. + q) ** 2
        s_scale_2 = 4 * w_t ** 2 / (1. + q) ** 2
        kwargs_1 = {'theta_E': 1., 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': 1., 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
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
        s_scale_1 = 4 * w_c ** 2 / (1. + q) ** 2
        s_scale_2 = 4 * w_t ** 2 / (1. + q) ** 2
        kwargs_1 = {'theta_E': 1., 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': 1., 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        f_x, f_y = self.chameleon.derivatives(x=x, y=1., **kwargs_light)
        f_x_1, f_y_1 = self.nie.derivatives(x=x, y=1., **kwargs_1)
        f_x_2, f_y_2 = self.nie.derivatives(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_x, (f_x_1 - f_x_2), decimal=5)
        npt.assert_almost_equal(f_y, (f_y_1 - f_y_2), decimal=5)

    def test_hessian(self):
        """

        :return:
        """
        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'theta_E': 1., 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        s_scale_1 = 4 * w_c ** 2 / (1. + q) ** 2
        s_scale_2 = 4 * w_t ** 2 / (1. + q) ** 2
        kwargs_1 = {'theta_E': 1., 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'theta_E': 1., 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        f_xx, f_yy, f_xy = self.chameleon.hessian(x=x, y=1., **kwargs_light)
        f_xx_1, f_yy_1, f_xy_1 = self.nie.hessian(x=x, y=1., **kwargs_1)
        f_xx_2, f_yy_2, f_xy_2 = self.nie.hessian(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(f_xx, (f_xx_1 - f_xx_2), decimal=5)
        npt.assert_almost_equal(f_yy, (f_yy_1 - f_yy_2), decimal=5)
        npt.assert_almost_equal(f_xy, (f_xy_1 - f_xy_2), decimal=5)


if __name__ == '__main__':
    pytest.main()

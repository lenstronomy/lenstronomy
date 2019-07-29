
import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.nie import NIE
from lenstronomy.LightModel.Profiles.chameleon import Chameleon, DoubleChameleon, TripleChameleon
import lenstronomy.Util.param_util as param_util


class TestChameleon(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_param_name(self):
        chameleon = Chameleon()
        names = chameleon.param_names
        assert names[0] == 'amp'

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        nie = NIE()

        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'amp': 1., 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        amp_new, w_c, w_t = chameleon._chameleonLens._theta_convert(1, w_c, w_t)
        kwargs_1 = {'amp': amp_new, 's_scale': s_scale_1, 'e1': e1, 'e2': e2}
        kwargs_2 = {'amp': amp_new, 's_scale': s_scale_2, 'e1': e1, 'e2': e2}
        flux = chameleon.function(x=x, y=1., **kwargs_light)
        flux1 = nie.function(x=x, y=1., **kwargs_1)
        flux2 = nie.function(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(flux, (flux1 - flux2) / (1. + q), decimal=5)


class TestDoubleChameleon(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == 'amp'

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        doublechameleon = DoubleChameleon()

        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.
        phi_G, q = 0.3, 0.8
        theta_E = 1.
        ratio = 2.
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'amp': 1., 'ratio': 2, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2, 'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2}

        kwargs_1 = {'amp': 1. / (1 + 1. / ratio), 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        kwargs_2 = {'amp': 1. / (1 + ratio), 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        flux = doublechameleon.function(x=x, y=1., **kwargs_light)
        flux1 = chameleon.function(x=x, y=1., **kwargs_1)
        flux2 = chameleon.function(x=x, y=1., **kwargs_2)
        npt.assert_almost_equal(flux, flux1 + flux2, decimal=8)


class TestTripleChameleon(object):
    """
    class to test the Moffat profile
    """
    def setup(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == 'amp'

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        triplechameleon = TripleChameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio12 = 2.
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {'amp': 1., 'ratio12': ratio12, 'ratio13': ratio13, 'w_c1': .5, 'w_t1': 1., 'e11': e1, 'e21': e2,
                        'w_c2': .1, 'w_t2': .5, 'e12': e1, 'e22': e2,
                        'w_c3': .1, 'w_t3': .5, 'e13': e1, 'e23': e2
                        }

        amp1 = 1. / (1. + 1. / ratio12 + 1. / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        kwargs_1 = {'amp': amp1, 'w_c': .5, 'w_t': 1., 'e1': e1, 'e2': e2}
        kwargs_2 = {'amp': amp2, 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        kwargs_3 = {'amp': amp3, 'w_c': .1, 'w_t': .5, 'e1': e1, 'e2': e2}
        flux = triplechameleon.function(x=x, y=1., **kwargs_light)
        flux1 = chameleon.function(x=x, y=1., **kwargs_1)
        flux2 = chameleon.function(x=x, y=1., **kwargs_2)
        flux3 = chameleon.function(x=x, y=1., **kwargs_3)
        npt.assert_almost_equal(flux, flux1 + flux2 + flux3, decimal=8)


if __name__ == '__main__':
    pytest.main()

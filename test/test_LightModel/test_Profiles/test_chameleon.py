import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.LightModel.Profiles.nie import NIE
from lenstronomy.LightModel.Profiles.chameleon import (
    Chameleon,
    DoubleChameleon,
    TripleChameleon,
)
from lenstronomy.LensModel.Profiles.chameleon import Chameleon as ChameleonLens
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util


class TestChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = Chameleon()
        names = chameleon.param_names
        assert names[0] == "amp"

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        nie = NIE()

        x = np.linspace(0.1, 10, 10)
        w_c, w_t = 0.5, 1.0
        phi_G, q = 0.3, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {"amp": 1.0, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        (
            amp_new,
            w_c,
            w_t,
            s_scale_1,
            s_scale_2,
        ) = chameleon._chameleonLens.param_convert(1, w_c, w_t, e1, e2)
        kwargs_1 = {"amp": amp_new, "s_scale": s_scale_1, "e1": e1, "e2": e2}
        kwargs_2 = {"amp": amp_new, "s_scale": s_scale_2, "e1": e1, "e2": e2}
        flux = chameleon.function(x=x, y=1.0, **kwargs_light)
        flux1 = nie.function(x=x, y=1.0, **kwargs_1)
        flux2 = nie.function(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(flux, flux1 - flux2, decimal=5)

    def test_lens_model_correspondence(self):
        """
        here we test the proportionality of the convergence of the lens model with the surface brightness of the light
        model
        """
        chameleon_lens = ChameleonLens()
        chameleon = Chameleon()

        x, y = util.make_grid(numPix=100, deltapix=0.1)
        e1, e2 = 0.0, 0
        w_c, w_t = 0.5, 1.0
        kwargs_light = {"amp": 1.0, "w_c": w_c, "w_t": w_t, "e1": e1, "e2": e2}
        kwargs_lens = {"alpha_1": 1.0, "w_c": w_c, "w_t": w_t, "e1": e1, "e2": e2}
        flux = chameleon.function(x=x, y=y, **kwargs_light)
        f_xx, f_xy, f_yx, f_yy = chameleon_lens.hessian(x=x, y=y, **kwargs_lens)
        kappa = 1 / 2.0 * (f_xx + f_yy)

        # flux2d = util.array2image(flux)
        # kappa2d = util.array2image(kappa)

        npt.assert_almost_equal(flux / np.mean(flux), kappa / np.mean(kappa), decimal=3)


class TestDoubleChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == "amp"

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        doublechameleon = DoubleChameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio = 2.0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_light = {
            "amp": 1.0,
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
            "amp": 1.0 / (1 + 1.0 / ratio),
            "w_c": 0.5,
            "w_t": 1.0,
            "e1": e1,
            "e2": e2,
        }
        kwargs_2 = {
            "amp": 1.0 / (1 + ratio),
            "w_c": 0.1,
            "w_t": 0.5,
            "e1": e1,
            "e2": e2,
        }
        flux = doublechameleon.function(x=x, y=1.0, **kwargs_light)
        flux1 = chameleon.function(x=x, y=1.0, **kwargs_1)
        flux2 = chameleon.function(x=x, y=1.0, **kwargs_2)
        npt.assert_almost_equal(flux, flux1 + flux2, decimal=8)


class TestTripleChameleon(object):
    """
    class to test the Moffat profile
    """

    def setup_method(self):
        pass

    def test_param_name(self):
        chameleon = DoubleChameleon()
        names = chameleon.param_names
        assert names[0] == "amp"

    def test_function(self):
        """

        :return:
        """
        chameleon = Chameleon()
        triplechameleon = TripleChameleon()

        x = np.linspace(0.1, 10, 10)
        phi_G, q = 0.3, 0.8
        ratio12 = 2.0
        ratio13 = 3
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
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
            "w_c3": 0.4,
            "w_t3": 0.8,
            "e13": e1,
            "e23": e2,
        }

        amp1 = 1.0 / (1.0 + 1.0 / ratio12 + 1.0 / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        kwargs_1 = {"amp": amp1, "w_c": 0.5, "w_t": 1.0, "e1": e1, "e2": e2}
        kwargs_2 = {"amp": amp2, "w_c": 0.1, "w_t": 0.5, "e1": e1, "e2": e2}
        kwargs_3 = {"amp": amp3, "w_c": 0.4, "w_t": 0.8, "e1": e1, "e2": e2}
        flux = triplechameleon.function(x=x, y=1.0, **kwargs_light)
        flux1 = chameleon.function(x=x, y=1.0, **kwargs_1)
        flux2 = chameleon.function(x=x, y=1.0, **kwargs_2)
        flux3 = chameleon.function(x=x, y=1.0, **kwargs_3)
        npt.assert_almost_equal(flux, flux1 + flux2 + flux3, decimal=8)


if __name__ == "__main__":
    pytest.main()

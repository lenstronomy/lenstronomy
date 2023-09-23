__author__ = "sibirrer"

import pytest
import numpy.testing as npt
import numpy as np
import scipy.integrate as integrate


class TestNumerics(object):
    """Tests the second derivatives of various lens models."""

    def setup_method(self):
        pass

    def assert_integrals(self, Model, kwargs):
        lightModel = Model()
        r = 2.0
        out = integrate.quad(
            lambda x: 2 * lightModel.light_3d(np.sqrt(x**2 + r**2), **kwargs),
            0,
            100,
        )
        light_2d_num = out[0]
        light_2d = lightModel.function(r, 0, **kwargs)
        npt.assert_almost_equal(light_2d_num / light_2d, 1.0, decimal=1)

    def test_PJaffe(self):
        kwargs = {"amp": 1.0, "Ra": 0.2, "Rs": 2.0}
        from lenstronomy.LightModel.Profiles.p_jaffe import PJaffe as Model

        self.assert_integrals(Model, kwargs)

    def test_hernquist(self):
        kwargs = {"amp": 1.0, "Rs": 5.0}
        from lenstronomy.LightModel.Profiles.hernquist import Hernquist as Model

        self.assert_integrals(Model, kwargs)

    def test_gaussian(self):
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian as Model

        kwargs = {"amp": 1.0 / 4.0, "sigma": 2.0}
        self.assert_integrals(Model, kwargs)

    def test_power_law(self):
        from lenstronomy.LightModel.Profiles.power_law import PowerLaw as Model

        kwargs = {"amp": 2, "gamma": 2, "e1": 0, "e2": 0}
        self.assert_integrals(Model, kwargs)

    def test_nie(self):
        from lenstronomy.LightModel.Profiles.nie import NIE as Model

        kwargs = {"amp": 2, "s_scale": 0.001, "e1": 0, "e2": 0}
        self.assert_integrals(Model, kwargs)
        kwargs = {"amp": 2, "s_scale": 1.0, "e1": 0, "e2": 0}
        self.assert_integrals(Model, kwargs)

    def test_chameleon(self):
        from lenstronomy.LightModel.Profiles.chameleon import Chameleon as Model

        kwargs = {"amp": 2, "w_c": 1, "w_t": 2, "e1": 0, "e2": 0}
        self.assert_integrals(Model, kwargs)

        from lenstronomy.LightModel.Profiles.chameleon import DoubleChameleon as Model

        kwargs = {
            "amp": 2,
            "ratio": 0.4,
            "w_c1": 1,
            "w_t1": 2,
            "e11": 0,
            "e21": 0,
            "w_c2": 2,
            "w_t2": 3,
            "e12": 0,
            "e22": 0,
        }
        self.assert_integrals(Model, kwargs)

        from lenstronomy.LightModel.Profiles.chameleon import TripleChameleon as Model

        kwargs = {
            "amp": 2,
            "ratio12": 0.4,
            "ratio13": 2,
            "w_c1": 1,
            "w_t1": 2,
            "e11": 0,
            "e21": 0,
            "w_c2": 2,
            "w_t2": 3,
            "e12": 0,
            "e22": 0,
            "w_c3": 0.2,
            "w_t3": 1,
            "e13": 0,
            "e23": 0,
        }
        self.assert_integrals(Model, kwargs)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

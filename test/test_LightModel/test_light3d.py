__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
import numpy as np
import scipy.integrate as integrate


class TestNumerics(object):
    """
    tests the second derivatives of various lens models
    """
    def setup(self):
        pass

    def assert_integrals(self, Model, kwargs):
        lightModel = Model()
        r = 2.
        out = integrate.quad(lambda x: 2 * lightModel.light_3d(np.sqrt(x ** 2 + r ** 2), **kwargs), 0, 100)
        light_2d_num = out[0]
        light_2d = lightModel.function(r, 0, **kwargs)
        npt.assert_almost_equal(light_2d_num/light_2d, 1., decimal=1)

    def test_PJaffe(self):
        kwargs = {'amp': 1., 'Ra': 0.2, 'Rs': 2.}
        from lenstronomy.LightModel.Profiles.p_jaffe import PJaffe as Model
        self.assert_integrals(Model, kwargs)

    def test_hernquist(self):
        kwargs = {'amp': 1.,  'Rs': 5.}
        from lenstronomy.LightModel.Profiles.hernquist import Hernquist as Model
        self.assert_integrals(Model, kwargs)

    def test_gaussian(self):
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian as Model
        kwargs = {'amp': 1. / 4., 'sigma_x': 2., 'sigma_y': 2.}
        self.assert_integrals(Model, kwargs)

    def test_power_law(self):
        from lenstronomy.LightModel.Profiles.power_law import PowerLaw as Model
        kwargs = {'amp': 2, 'gamma': 2, 'e1': 0, 'e2': 0}
        self.assert_integrals(Model, kwargs)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

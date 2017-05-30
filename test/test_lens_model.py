__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.ImSim.lens_model import LensModel


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.kwargs_options = {'lens_model_list': ['GAUSSIAN']}

        self.lensModel = LensModel(self.kwargs_options)
        self.kwargs = [{'amp': 1., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}]

    def test_mass(self):
        output = self.lensModel.mass(x=1., y=1., sigma_crit=1.9e+15, kwargs=self.kwargs)
        npt.assert_almost_equal(output, -11039296368203.469, decimal=5)

    def test_kappa(self):
        output = self.lensModel.kappa(x=1., y=1., kwargs=self.kwargs)
        assert output == -0.0058101559832649833

    def test_potential(self):
        output = self.lensModel.potential(x=1., y=1., kwargs=self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1., y=1., kwargs=self.kwargs)
        assert output1 == -0.19470019576785122/(8*np.pi)
        assert output2 == -0.19470019576785122/(8*np.pi)

    def test_gamma(self):
        output1, output2 = self.lensModel.gamma(x=1., y=1., kwargs=self.kwargs)
        assert output1 == 0
        assert output2 == 0.048675048941962805/(8*np.pi)

    def test_magnification(self):
        output = self.lensModel.magnification(x=1., y=1., kwargs=self.kwargs)
        assert output == 0.98848384784633392

    def test_all(self):
        potential, alpha1, alpha2, kappa, gamma1, gamma2, mag = self.lensModel.all(x=1., y=1., kwargs=self.kwargs)
        assert potential == 0.77880078307140488/(8*np.pi)
        assert alpha1 == -0.19470019576785122/(8*np.pi)
        assert alpha2 == -0.19470019576785122/(8*np.pi)
        assert kappa == -0.0058101559832649833
        assert gamma1 == 0
        assert gamma2 == 0.048675048941962805/(8*np.pi)
        assert mag == 0.98848384784633392

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1., y=1., kwargs=self.kwargs)
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

if __name__ == '__main__':
    pytest.main("-k TestLensModel")
__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.single_plane import SinglePlane


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.lensModel = SinglePlane(['GAUSSIAN'])
        self.kwargs = [{'amp': 1., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}]

    def test_potential(self):
        output = self.lensModel.potential(x=1., y=1., kwargs=self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1., y=1., kwargs=self.kwargs)
        assert output1 == -0.19470019576785122/(8*np.pi)
        assert output2 == -0.19470019576785122/(8*np.pi)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1., y=1., kwargs=self.kwargs)
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

    def test_mass_2d(self):
        lensModel = SinglePlane(['GAUSSIAN_KAPPA'])
        output = lensModel.mass_2d(r=1, kwargs=self.kwargs)
        assert output == 0.11750309741540453


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
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
        kwargs = [{'amp': 1., 'sigma': 2., 'center_x': 0., 'center_y': 0.}]
        output = lensModel.mass_2d(r=1, kwargs=kwargs)
        assert output == 0.11750309741540453

    def test_bool_list(self):
        lensModel = SinglePlane(['SPEMD', 'SHEAR'])
        kwargs = [{'theta_E': 1, 'gamma': 1, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0},
                           {'e1': 0.01, 'e2': -0.02}]
        alphax_1, alphay_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_list, alphay_1_list = lensModel.alpha(1, 1, kwargs, k=[0])
        npt.assert_almost_equal(alphax_1, alphax_1_list, decimal=5)
        npt.assert_almost_equal(alphay_1, alphay_1_list, decimal=5)

        alphax_1_1, alphay_1_1 = lensModel.alpha(1, 1, kwargs, k=0)
        alphax_1_2, alphay_1_2 = lensModel.alpha(1, 1, kwargs, k=1)
        alphax_full, alphay_full = lensModel.alpha(1, 1, kwargs, k=None)
        npt.assert_almost_equal(alphax_1_1 + alphax_1_2, alphax_full, decimal=5)
        npt.assert_almost_equal(alphay_1_1 + alphay_1_2, alphay_full, decimal=5)

    def test_init(self):
        lens_model_list = ['TNFW', 'SPEMD_SMOOTH', 'TRIPLE_CHAMELEON']
        lensModel = SinglePlane(lens_model_list=lens_model_list)
        assert lensModel.func_list[0].param_names[0] == 'Rs'


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

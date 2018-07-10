__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.lens_model import LensModel


class TestLensModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.lensModel = LensModel(['GAUSSIAN'])
        self.kwargs = [{'amp': 1., 'sigma_x': 2., 'sigma_y': 2., 'center_x': 0., 'center_y': 0.}]

    def test_init(self):
        lens_model_list = ['FLEXION', 'SIS_TRUNCATED', 'SERSIC', 'SERSIC_ELLIPSE',
                           'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE', 'INTERPOL', 'INTERPOL_SCALED',
                           'SHAPELETS_POLAR', 'DIPOLE', 'GAUSSIAN_KAPPA_ELLIPSE', 'MULTI_GAUSSIAN_KAPPA'
                            , 'MULTI_GAUSSIAN_KAPPA_ELLIPSE', 'CHAMELEON', 'DOUBLE_CHAMELEON']
        lensModel = LensModel(lens_model_list)
        assert len(lensModel.lens_model_list) == len(lens_model_list)

    def test_mass(self):
        output = self.lensModel.mass(x=1., y=1., epsilon_crit=1.9e+15, kwargs=self.kwargs)
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

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1., y=1., kwargs=self.kwargs)
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

    def test_mass_2d(self):
        lensModel = LensModel(['GAUSSIAN_KAPPA'])
        kwargs = [{'amp': 1., 'sigma': 2.}]
        output = lensModel.mass_2d(r=1, kwargs=kwargs)
        assert output == 0.11750309741540453

    def test_param_name_list(self):
        lens_model_list = ['FLEXION', 'SIS_TRUNCATED', 'SERSIC', 'SERSIC_ELLIPSE',
                           'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE', 'INTERPOL', 'INTERPOL_SCALED',
                           'SHAPELETS_POLAR', 'DIPOLE', 'GAUSSIAN_KAPPA_ELLIPSE', 'MULTI_GAUSSIAN_KAPPA'
            , 'MULTI_GAUSSIAN_KAPPA_ELLIPSE']
        lensModel = LensModel(lens_model_list)
        param_name_list = lensModel.param_name_list()
        assert len(lens_model_list) == len(param_name_list)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")
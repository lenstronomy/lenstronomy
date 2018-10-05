__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.nfw import NFW


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

        lens_model_list = ['NFW']
        lensModel = LensModel(lens_model_list, interpol = True, lookup=True)
        x,y = 0.2,1
        kwargs = [{'theta_Rs':1, 'Rs': 0.5, 'center_x':0, 'center_y':0}]
        value = lensModel.potential(x,y,kwargs)
        nfw_interp = NFW(interpol=True, lookup=True)
        value_interp_lookup = nfw_interp.function(x, y, **kwargs[0])
        npt.assert_almost_equal(value, value_interp_lookup, decimal=4)

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

    def test_arrival_time(self):
        z_lens = 0.5
        z_source = 1.5
        x_image, y_image = 1., 0
        lensModel = LensModel(lens_model_list=['SIS'], multi_plane=True, redshift_list=[z_lens], z_source=z_source)
        kwargs = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        arrival_time_mp = lensModel.arrival_time(x_image, y_image, kwargs)
        lensModel_sp = LensModel(lens_model_list=['SIS'], z_source=z_source, z_lens=z_lens)
        arrival_time_sp = lensModel_sp.arrival_time(x_image, y_image, kwargs)
        npt.assert_almost_equal(arrival_time_sp, arrival_time_mp, decimal=8)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

__author__ = 'sibirrer'

import numpy as np
import pytest

from lenstronomy.LightModel.light_model import LightModel


class TestLensLightModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.lensLightModel = LightModel(light_model_list=['GAUSSIAN'])
        self.kwargs = [{'amp': 1., 'center_x': 0., 'center_y': 0., 'sigma_x': 2., 'sigma_y': 2.}]

    def test_init(self):
        model_list = ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC', 'BULDGE_DISK', 'SHAPELETS', 'UNIFORM']
        lightModel = LightModel(light_model_list=model_list)
        assert len(lightModel.profile_type_list) == len(model_list)

    def test_surface_brightness(self):
        output = self.lensLightModel.surface_brightness(x=1, y=1, kwargs_list=self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)

    def test_functions_split(self):
        output = self.lensLightModel.functions_split(x=1, y=1, kwargs_list=self.kwargs)
        assert output[0][0] == 0.77880078307140488/(8*np.pi)

    def test_re_normalize_flux(self):
        kwargs_out = self.lensLightModel.re_normalize_flux(kwargs_list=self.kwargs, norm_factor=2)
        assert kwargs_out[0]['amp'] == 2 * self.kwargs[0]['amp']


if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LightModel.light_model import LightModel


class TestLightModel(object):
    """
    tests the source model routines
    """

    def setup(self):
        self.light_model_list = ['GAUSSIAN', 'MULTI_GAUSSIAN', 'SERSIC', 'SERSIC_ELLIPSE', 'DOUBLE_SERSIC',
                                 'CORE_SERSIC', 'DOUBLE_CORE_SERSIC', 'BULDGE_DISK', 'SHAPELETS', 'HERNQUIST',
                                 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE', 'UNIFORM', 'NONE'
                                 ]
        self.kwargs = [
            {'amp': 1., 'sigma_x': 1, 'sigma_y': 1., 'center_x': 0, 'center_y': 0},  # 'GAUSSIAN'
            {'amp': [1., 2], 'sigma': [1, 3], 'center_x': 0, 'center_y': 0},  # 'MULTI_GAUSSIAN'
            {'I0_sersic': 1, 'R_sersic': 0.5, 'n_sersic': 1, 'center_x': 0, 'center_y': 0},  # 'SERSIC'
            {'I0_sersic': 1, 'R_sersic': 0.5, 'n_sersic': 1, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0, 'center_y': 0},  # 'SERSIC_ELLIPSE'
            {'I0_sersic': 1, 'R_sersic': 0.5, 'n_sersic': 1, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0, 'center_y': 0,
             'I0_2': 1, 'R_2': 0.05, 'n_2': 2, 'phi_G_2': 0, 'q_2': 1},  # 'DOUBLE_SERSIC'
            {'I0_sersic': 1, 'R_sersic': 0.5, 'Re': 0.1, 'gamma': 2., 'n_sersic': 1, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0, 'center_y': 0},
            # 'CORE_SERSIC'
            {'I0_sersic': 1, 'R_sersic': 0.5, 'Re': 0.1, 'gamma': 2., 'n_sersic': 1, 'q': 0.8, 'phi_G': 0.5,
             'center_x': 0, 'center_y': 0, 'I0_2': 1, 'R_2': 0.05, 'n_2': 2, 'phi_G_2': 0, 'q_2': 1},
            # 'DOUBLE_CORE_SERSIC'
            {'I0_b': 1, 'R_b': 0.1, 'phi_G_b': 0, 'q_b': 1, 'I0_d': 2, 'R_d': 1, 'phi_G_d': 0.5, 'q_d': 0.7, 'center_x': 0, 'center_y': 0},  # BULDGE_DISK
            {'amp': [1, 1, 1], 'beta': 0.5, 'n_max': 1, 'center_x': 0, 'center_y': 0},  # 'SHAPELETS'
            {'sigma0': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0},  # 'HERNQUIST'
            {'sigma0': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0},  # 'HERNQUIST_ELLIPSE'
            {'sigma0': 1, 'Ra': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0},  # 'PJAFFE'
            {'sigma0': 1, 'Ra': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0},  # 'PJAFFE_ELLIPSE'
            {'mean': 1},  # 'UNIFORM'
            {}]# 'NONE'

        self.LightModel = LightModel(light_model_list=self.light_model_list)

    def test_init(self):
        model_list = ['CORE_SERSIC', 'DOUBLE_CORE_SERSIC', 'BULDGE_DISK', 'SHAPELETS', 'UNIFORM']
        lightModel = LightModel(light_model_list=model_list)
        assert len(lightModel.profile_type_list) == len(model_list)

    def test_surface_brightness(self):
        output = self.LightModel.surface_brightness(x=1, y=1, kwargs_list=self.kwargs)
        npt.assert_almost_equal(output, 2.544428612985992, decimal=6)

    def test_surface_brightness_array(self):
        output = self.LightModel.surface_brightness(x=[1], y=[1], kwargs_list=self.kwargs)
        npt.assert_almost_equal(output[0], 2.544428612985992, decimal=6)

    def test_functions_split(self):
        output = self.LightModel.functions_split(x=1., y=1., kwargs_list=self.kwargs)
        assert output[0][0] == 0.058549831524319168

    def test_re_normalize_flux(self):
        kwargs_out = self.LightModel.re_normalize_flux(kwargs_list=self.kwargs, norm_factor=2)
        assert kwargs_out[0]['amp'] == 2 * self.kwargs[0]['amp']


if __name__ == '__main__':
    pytest.main()
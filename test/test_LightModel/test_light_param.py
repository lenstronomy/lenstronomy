__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.LightModel.light_param import LightParam


class TestParam(object):

    def setup(self):
        self.light_model_list = ['GAUSSIAN', 'MULTI_GAUSSIAN', 'SERSIC', 'SERSIC_ELLIPSE',
                                 'CORE_SERSIC', 'SHAPELETS', 'HERNQUIST',
                                 'HERNQUIST_ELLIPSE', 'PJAFFE', 'PJAFFE_ELLIPSE', 'UNIFORM'
                                 ]
        self.kwargs = [
            {'amp': 1., 'sigma_x': 1, 'sigma_y': 1., 'center_x': 0, 'center_y': 0},  # 'GAUSSIAN'
            {'amp': [1., 2], 'sigma': [1, 3], 'center_x': 0, 'center_y': 0},  # 'MULTI_GAUSSIAN'
            {'amp': 1, 'R_sersic': 0.5, 'n_sersic': 1, 'center_x': 0, 'center_y': 0},  # 'SERSIC'
            {'amp': 1, 'R_sersic': 0.5, 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0, 'center_y': 0},  # 'SERSIC_ELLIPSE'
            {'amp': 1, 'R_sersic': 0.5, 'Re': 0.1, 'gamma': 2., 'n_sersic': 1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0, 'center_y': 0},
            # 'CORE_SERSIC'
            {'amp': [1, 1, 1], 'beta': 0.5, 'n_max': 1, 'center_x': 0, 'center_y': 0},  # 'SHAPELETS'
            {'amp': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0},  # 'HERNQUIST'
            {'amp': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1},  # 'HERNQUIST_ELLIPSE'
            {'amp': 1, 'Ra': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0},  # 'PJAFFE'
            {'amp': 1, 'Ra': 1, 'Rs': 0.5, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1},  # 'PJAFFE_ELLIPSE'
            {'amp': 1},  # 'UNIFORM'

        ]
        self.kwargs_sigma = [
            {'amp_sigma': 1., 'sigma_x_sigma': 1, 'sigma_y_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},
            # 'GAUSSIAN'
            {'amp_sigma': [1., 1.], 'sigma_sigma': [1, 1], 'center_x_sigma': 0, 'center_y_sigma': 0},
            # 'MULTI_GAUSSIAN'
            {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1, 'center_y_sigma': 1},  # 'SERSIC'
            {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'SERSIC_ELLIPSE'
            {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'e1_sigma': 0.1, 'e2_sigma': 0.1, 'Re_sigma': 0.01, 'gamma_sigma': 0.1},  # 'CORE_SERSIC'
            {'amp_sigma': [1, 1, 1], 'beta_sigma': 0.1, 'n_max_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS'
            {'amp_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'HERNQUIST'
            {'amp_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'HERNQUIST_ELLIPSE'
            {'amp_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'PJAFFE'
            {'amp_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'PJAFFE'
            {'amp_sigma': 0.1},  # 'UNIFORM'

        ]
        self.kwargs_fixed = [{}, {'sigma': [1, 3]}, {}, {}, {}, {'n_max': 1}, {}, {}, {}, {}, {}
                             ]
        self.kwargs_fixed_linear = [{}, {'sigma': [1, 3]}, {}, {}, {}, {'n_max': 1}, {}, {}, {}, {}, {}]
        self.kwargs_mean = []
        for i in range(len(self.light_model_list)):
            kwargs_mean_k = self.kwargs[i].copy()
            kwargs_mean_k.update(self.kwargs_sigma[i])
            self.kwargs_mean.append(kwargs_mean_k)
        self.param = LightParam(light_model_list=self.light_model_list,
                               kwargs_fixed=self.kwargs_fixed, type='source_light', linear_solver=False)
        self.param_linear = LightParam(light_model_list=self.light_model_list,
                                kwargs_fixed=self.kwargs_fixed_linear, type='source_light', linear_solver=True)
        self.param_fixed = LightParam(light_model_list=self.light_model_list,
                                kwargs_fixed=self.kwargs, type='source_light', linear_solver=False)

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        print(self.param.kwargs_fixed, 'test')
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_fixed.setParams(self.kwargs)
        kwargs_new, _ = self.param_fixed.getParams(args, i=0)
        args_new = self.param_fixed.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_linear.setParams(self.kwargs)
        kwargs_new, _ = self.param_linear.getParams(args, i=0)
        args_new = self.param_linear.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 59

    def test_num_param_linear(self):

        kwargs_fixed = [{}, {'sigma': [1, 3]}, {}, {}, {}, {'n_max': 1}, {}, {},
                             {}, {}, {}]
        param = LightParam(light_model_list=self.light_model_list,
                                kwargs_fixed=kwargs_fixed, type='source_light', linear_solver=True)

        num = param.num_param_linear()
        assert num == 14


if __name__ == '__main__':
    pytest.main()

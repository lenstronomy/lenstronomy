__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.LightModel.light_param import LightParam


class TestParam(object):

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
            {},  # 'NONE'

        ]
        self.kwargs_sigma = [
            {'amp_sigma': 1., 'sigma_x_sigma': 1, 'sigma_y_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},
            # 'GAUSSIAN'
            {'amp_sigma': 1., 'sigma_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},
            # 'MULTI_GAUSSIAN'
            {'I0_sersic_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1, 'center_y_sigma': 1},  # 'SERSIC'
            {'I0_sersic_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'ellipse_sigma': 0.1},  # 'SERSIC_ELLIPSE'
            {'I0_sersic_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'ellipse_sigma': 0.1, 'I0_2_sigma': 1, 'R_2_sigma': 0.05, 'n_2_sigma': 2},  # 'DOUBLE_SERSIC'
            {'I0_sersic_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'ellipse_sigma': 0.1, 'Re_sigma': 0.01, 'gamma_sigma': 0.1},  # 'CORE_SERSIC'
            {'I0_sersic_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
             'center_y_sigma': 1, 'ellipse_sigma': 0.1, 'Re_sigma': 0.01, 'gamma_sigma': 0.1, 'I0_2_sigma': 1, 'R_2_sigma': 0.05, 'n_2_sigma': 2},  # 'DOUBLE_CORE_SERSIC'
            {'I0_b_sigma': 1, 'R_b_sigma': 0.1, 'ellipse_sigma': 0.1, 'I0_d_sigma': 2, 'R_d_sigma': 1,
             'center_x_sigma': 0, 'center_y_sigma': 0},  # BULDGE_DISK
            {'amp_sigma': 1, 'beta_sigma': 0.1, 'n_max_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS'
            {'sigma0_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'HERNQUIST'
            {'sigma0_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'ellipse_sigma': 0.1},  # 'HERNQUIST_ELLIPSE'
            {'sigma0_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'PJAFFE'
            {'sigma0_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'ellipse_sigma': 0.1},  # 'PJAFFE'
            {'mean_sigma': 0.1},  # 'UNIFORM'
            {},  # 'NONE'

        ]
        self.kwargs_fixed = [{}, {'sigma': [1, 3]}, {}, {}, {}, {}, {}, {}, {'amp': [1, 1, 1]}, {}, {}, {}, {}, {}, {}
                             ]
        self.kwargs_mean = []
        for i in range(len(self.light_model_list)):
            kwargs_mean_k = self.kwargs[i].copy()
            kwargs_mean_k.update(self.kwargs_sigma[i])
            self.kwargs_mean.append(kwargs_mean_k)
        self.param = LightParam(light_model_list=self.light_model_list,
                               kwargs_fixed=self.kwargs_fixed, type='source_light', linear_solver=False)
        self.param_fixed = LightParam(light_model_list=self.light_model_list,
                                kwargs_fixed=self.kwargs, type='source_light', linear_solver=False)

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_fixed.setParams(self.kwargs)
        kwargs_new, _ = self.param_fixed.getParams(args, i=0)
        args_new = self.param_fixed.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_param_init(self):
        mean, sigma = self.param.param_init(self.kwargs_mean)
        assert mean[0] == 0

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 93


if __name__ == '__main__':
    pytest.main()

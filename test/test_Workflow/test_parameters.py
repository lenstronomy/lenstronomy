__author__ = 'sibirrer'

import numpy as np
import pytest

from lenstronomy.Workflow.parameters import Param, ParamUpdate


class TestParam(object):

    def setup(self):
        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                          'lens_light_model_list': ['DOUBLE_SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_param = {}
        kwargs_fixed_lens = [{'gamma': 1.9}] #for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x':0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        self.param_class = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                 kwargs_fixed_lens_light, kwargs_fixed_ps)

    def test_num_param(self):
        num_param, list = self.param_class.num_param()
        assert list[0] == 'theta_E'
        assert num_param == 15

        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                        'lens_light_model_list': ['DOUBLE_SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_param = {}
        kwargs_fixed_lens = [{'gamma': 1.9}]  # for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x': 0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        param_class_linear = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                        kwargs_fixed_lens_light, kwargs_fixed_ps, linear_solver=False)
        num_param, list = param_class_linear.num_param()
        assert list[0] == 'theta_E'
        assert num_param == 18

    def test_get_params(self):
        kwargs_true_lens = [{'theta_E': 1.,'gamma':1.9,'q':0.8,'phi_G':1.5, 'center_x':0., 'center_y':0.}] #for SPEP lens
        kwargs_true_source = [{'amp': 1*2*np.pi*0.1**2,'center_x':0.2, 'center_y':0.2, 'sigma_x': 0.1, 'sigma_y': 0.1}]
        kwargs_true_lens_light = [{'center_x_2': 0.1, 'n_2': 1, 'center_x': -0.06, 'center_y': 0.4, 'phi_G': 4.8,
                                  'q': 0.86, 'R_2': 1.2, 'I0_2': 1.7, 'center_y_2': 0.14, 'n_sersic': 1.7,
                                  'I0_sersic': 11.8, 'R_sersic': 0.697, 'phi_G_2': 0, 'q_2': 1}]
        kwargs_true_ps = [{'point_amp': [1, 1], 'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        args = self.param_class.setParams(kwargs_true_lens, kwargs_true_source, kwargs_lens_light=kwargs_true_lens_light, kwargs_ps=kwargs_true_ps)
        lens_dict_list, source_dict, lens_light_dic, else_dict = self.param_class.getParams(args)
        lens_dict = lens_dict_list[0]
        assert lens_dict['theta_E'] == 1.
        assert lens_dict['gamma'] == 1.9
        assert lens_dict['q'] == 0.8
        assert lens_dict['phi_G'] == 1.5
        assert lens_dict['center_x'] == 0.
        assert lens_dict['center_y'] == 0.
        assert lens_light_dic[0]['center_x'] == -0.06

    def test_param_init(self):
        kwargs_mean_lens = [{'theta_E': 1., 'theta_E_sigma': 0.1, 'gamma':1.9, 'gamma_sigma': 0.2 ,'q':0.8,'phi_G':1.5, 'ellipse_sigma': 0.2, 'center_x':0., 'center_y':0., 'center_x_sigma':0., 'center_y_sigma':0.}] #for SPEP lens
        kwargs_mean_source = [{'amp': 1*2*np.pi*0.1**2, 'amp_sigma': 1, 'center_x': 0.2, 'center_y': 0.2, 'center_x_sigma': 0.2, 'center_y_sigma': 0.2, 'sigma_x': 0.1, 'sigma_y': 0.1, 'sigma_x_sigma': 0.1, 'sigma_y_sigma': 0.1}]
        kwargs_mean_lens_light = [{'n_2': 1, 'n_2_sigma': 0.1, 'center_x': -0.06, 'center_y': 0.4, 'center_x_sigma': -0.06, 'center_y_sigma': 0.4, 'phi_G': 4.8,
                                  'q': 0.86, 'ellipse_sigma': 0.2, 'R_2': 1.2, 'R_2_sigma': 0.1, 'I0_2': 1.7, 'I0_2_sigma': 1, 'n_sersic': 1.7, 'n_sersic_sigma': 1,
                                  'I0_sersic': 11.8, 'I0_sersic_sigma': 1, 'R_sersic': 0.697, 'R_sersic_sigma': 0.1, 'phi_G_2': 0, 'q_2': 1}]
        kwargs_mean_ps = [{'point_amp': [1, 1], 'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        mean, sigma = self.param_class.param_init(kwargs_mean_lens, kwargs_mean_source, kwargs_mean_lens_light, kwargs_mean_ps)
        assert mean[0] == 1
        assert sigma[0] == 0.1

    def test_add_fixed_source(self):
        kwargs_fixed = [{}]
        kwargs_fixed = self.param_class._add_fixed_source(kwargs_fixed)
        assert 1 == 1


class TestParamUpdate(object):
    def setup(self):
        kwargs_fixed_lens = [{}, {}]
        kwargs_fixed_source = [{}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_ps = [{}]
        self.paramUpdate = ParamUpdate(kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps)

    def test_update_fixed_simple(self):
        kwargs_lens = [{'theta_E': 1, 'gamma': 2}, {}]
        kwargs_source = [{'test_source': 1}]
        kwargs_lens_light = [{'test_lens_light': 1}]
        kwargs_ps = [{'test_point_source': 1}]
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps = self.paramUpdate.update_fixed_simple(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, fix_lens=True,
                             fix_source=True, fix_lens_light=True, fix_point_source=True, gamma_fixed=True)
        assert kwargs_fixed_lens[0]['gamma'] == 2
        assert kwargs_fixed_source[0]['test_source'] == 1
        assert kwargs_fixed_lens_light[0]['test_lens_light'] == 1
        assert kwargs_fixed_ps[0]['test_point_source'] == 1


if __name__ == '__main__':
    pytest.main()

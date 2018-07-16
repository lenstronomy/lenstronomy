__author__ = 'sibirrer'

import numpy as np
import pytest

from lenstronomy.Workflow.parameters import Param, ParamUpdate


class TestParam(object):

    def setup(self):
        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                          'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_param = {}
        kwargs_fixed_lens = [{'gamma': 1.9}] #for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x':0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_cosmo = [{}]
        self.param_class = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                 kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo)

    def test_num_param(self):
        num_param, list = self.param_class.num_param()
        assert list[0] == 'theta_E_lens'
        assert num_param == 9

        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                        'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_param = {}
        kwargs_fixed_lens = [{'gamma': 1.9}]  # for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x': 0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_cosmo = [{}]
        param_class_linear = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                        kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo, linear_solver=False)
        num_param, list = param_class_linear.num_param()
        assert list[0] == 'theta_E_lens'
        assert num_param == 11

    def test_get_params(self):
        kwargs_true_lens = [{'theta_E': 1.,'gamma':1.9, 'e1':0.01, 'e2':-0.01, 'center_x':0., 'center_y':0.}] #for SPEP lens
        kwargs_true_source = [{'amp': 1*2*np.pi*0.1**2,'center_x':0.2, 'center_y':0.2, 'sigma_x': 0.1, 'sigma_y': 0.1}]
        kwargs_true_lens_light = [{'center_x': -0.06, 'center_y': 0.4, 'phi_G': 4.8,
                                  'q': 0.86, 'n_sersic': 1.7,
                                  'amp': 11.8, 'R_sersic': 0.697, 'phi_G_2': 0}]
        kwargs_true_ps = [{'point_amp': [1, 1], 'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_cosmo = [{}]
        args = self.param_class.setParams(kwargs_true_lens, kwargs_true_source, kwargs_lens_light=kwargs_true_lens_light, kwargs_ps=kwargs_true_ps, kwargs_cosmo=kwargs_cosmo)
        lens_dict_list, source_dict, lens_light_dic, ps_dict, cosmos_dict = self.param_class.getParams(args)
        lens_dict = lens_dict_list[0]
        assert lens_dict['theta_E'] == 1.
        assert lens_dict['gamma'] == 1.9
        assert lens_dict['e1'] == 0.01
        assert lens_dict['e2'] == -0.01
        assert lens_dict['center_x'] == 0.
        assert lens_dict['center_y'] == 0.
        assert lens_light_dic[0]['center_x'] == -0.06

    def test_param_init(self):
        kwargs_mean_lens = [{'theta_E': 1., 'theta_E_sigma': 0.1, 'gamma':1.9, 'gamma_sigma': 0.2, 'e1':0.01, 'e2':-0.0, 'e1_sigma': 0.2, 'e2_sigma': 0.2, 'center_x':0., 'center_y':0., 'center_x_sigma':0., 'center_y_sigma':0.}] #for SPEP lens
        kwargs_mean_source = [{'amp': 1*2*np.pi*0.1**2, 'amp_sigma': 1, 'center_x': 0.2, 'center_y': 0.2, 'center_x_sigma': 0.2, 'center_y_sigma': 0.2, 'sigma_x': 0.1, 'sigma_y': 0.1, 'sigma_x_sigma': 0.1, 'sigma_y_sigma': 0.1}]
        kwargs_mean_lens_light = [{'center_x': -0.06, 'center_y': 0.4, 'center_x_sigma': -0.06, 'center_y_sigma': 0.4, 'e1':0.01, 'e2':-0.0, 'ellipse_sigma': 0.2, 'n_sersic': 1.7, 'n_sersic_sigma': 1,
                                  'amp': 11.8, 'amp_sigma': 1, 'R_sersic': 0.697, 'R_sersic_sigma': 0.1, 'phi_G_2': 0}]
        kwargs_mean_ps = [{'point_amp': [1, 1], 'ra_image': [-1, 1], 'dec_image': [-1, 1]}]

        mean, sigma = self.param_class.param_init(kwargs_mean_lens, kwargs_mean_source, kwargs_mean_lens_light, kwargs_mean_ps, kwargs_mean_cosmo=None)
        assert mean[0] == 1
        assert sigma[0] == 0.1

    def test_add_fixed_source(self):
        kwargs_fixed = [{}]
        kwargs_fixed = self.param_class._add_fixed_source(kwargs_fixed)
        assert 1 == 1

    def test_get_cosmo(self):
        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                        'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION'],
                        'cosmo_type': 'D_dt'}
        kwargs_param = {}
        kwargs_fixed_lens = [{'gamma': 1.9}]  # for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x': 0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_cosmo = {'D_dt': 1000}
        param_class = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                 kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo)

        kwargs_true_lens = [
            {'theta_E': 1., 'gamma': 1.9, 'e1':0.01, 'e2':-0.01, 'center_x': 0., 'center_y': 0.}]  # for SPEP lens
        kwargs_true_source = [
            {'amp': 1 * 2 * np.pi * 0.1 ** 2, 'center_x': 0.2, 'center_y': 0.2, 'sigma_x': 0.1, 'sigma_y': 0.1}]
        kwargs_true_lens_light = [{'center_x': -0.06, 'center_y': 0.4, 'phi_G': 4.8,
                                   'q': 0.86, 'n_sersic': 1.7,
                                   'amp': 11.8, 'R_sersic': 0.697, 'phi_G_2': 0}]
        kwargs_true_ps = [{'point_amp': [1, 1], 'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        args = param_class.setParams(kwargs_true_lens, kwargs_true_source,
                                          kwargs_lens_light=kwargs_true_lens_light, kwargs_ps=kwargs_true_ps,
                                          kwargs_cosmo={'D_dt': 1000})
        assert param_class.cosmoParams._Ddt_sampling is True

    def test_mass_scaling(self):
        kwargs_model = {'lens_model_list': ['SIS', 'NFW', 'NFW']}
        kwargs_constraints = {'mass_scaling': True, 'mass_scaling_list': [False, 0, 0], 'num_scale_factor': 1}
        kwargs_fixed_lens = [{}, {'theta_Rs': 0.1}, {'theta_Rs': 0.3}]
        kwargs_fixed_cosmo = {}
        param_class = Param(kwargs_model, kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_cosmo=kwargs_fixed_cosmo)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0},
                       {'theta_Rs': 0.1, 'Rs': 5, 'center_x': 1., 'center_y': 0},
                       {'theta_Rs': 0.1, 'Rs': 5, 'center_x': 0, 'center_y': 1.}]
        kwargs_source = []
        kwargs_lens_light = []
        kwargs_ps = []
        mass_scale = 2
        kwargs_cosmo = {'scale_factor': [mass_scale]}
        args = param_class.setParams(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo=kwargs_cosmo)
        assert args[-1] == mass_scale

        kwargs_lens, _, _, _, _ = param_class.getParams(args)
        assert kwargs_lens[0]['theta_E'] == 1
        assert kwargs_lens[1]['theta_Rs'] == 0.1 * mass_scale
        assert kwargs_lens[2]['theta_Rs'] == 0.3 * mass_scale

        kwargs_lens, _, _, _, _ = param_class.getParams(args, bijective=True)
        assert kwargs_lens[0]['theta_E'] == 1
        assert kwargs_lens[1]['theta_Rs'] == 0.1
        assert kwargs_lens[2]['theta_Rs'] == 0.3

    def test_joint_with_light(self):
        kwargs_model = {'lens_model_list': ['CHAMELEON'], 'lens_light_model_list': ['CHAMELEON']}
        kwargs_constraints = {'joint_with_light_list': [0]}
        kwargs_lens = [{'theta_E': 10}]
        kwargs_lens_light = [{'amp': 1, 'w_t': 0.5, 'w_c': 0.1, 'center_x': 0, 'center_y': 0.3, 'e1': 0.1, 'e2': -0.2}]
        param = Param(kwargs_model=kwargs_model, kwargs_constraints=kwargs_constraints)
        args = param.setParams(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light)
        kwargs_lens_out, _, kwargs_lens_light_out, _, _ = param.getParams(args)
        assert kwargs_lens_out[0]['w_c'] == kwargs_lens_light[0]['w_c']
        assert kwargs_lens_light_out[0]['w_c'] == kwargs_lens_light[0]['w_c']


class TestParamUpdate(object):
    def setup(self):
        kwargs_fixed_lens = [{}, {}]
        kwargs_fixed_source = [{}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_ps = [{}]
        kwargs_fixed_cosmo = [{}]
        self.paramUpdate = ParamUpdate(kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo)

    def test_update_fixed_simple(self):
        kwargs_lens = [{'theta_E': 1, 'gamma': 2}, {}]
        kwargs_source = [{'test_source': 1}]
        kwargs_lens_light = [{'test_lens_light': 1}]
        kwargs_ps = [{'test_point_source': 1}]
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo = self.paramUpdate.update_fixed_simple(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo={}, fix_lens=True,
                             fix_source=True, fix_lens_light=True, fix_point_source=True)
        assert kwargs_fixed_lens[0]['gamma'] == 2
        assert kwargs_fixed_source[0]['test_source'] == 1
        assert kwargs_fixed_lens_light[0]['test_lens_light'] == 1
        assert kwargs_fixed_ps[0]['test_point_source'] == 1


if __name__ == '__main__':
    pytest.main()

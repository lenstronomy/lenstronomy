__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Sampling.parameters import Param


class TestParam(object):

    def setup(self):
        kwargs_model = {'lens_model_list': ['SPEP'], 'source_light_model_list': ['GAUSSIAN'],
                          'lens_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        kwargs_param = {'num_point_source_list': [2]}
        kwargs_fixed_lens = [{'gamma': 1.9}] #for SPEP lens
        kwargs_fixed_source = [{'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x':0.2, 'center_y': 0.2}]
        kwargs_fixed_ps = [{'ra_image': [-1, 1], 'dec_image': [-1, 1]}]
        kwargs_fixed_lens_light = [{}]
        kwargs_fixed_cosmo = [{}]
        self.param_class = Param(kwargs_model, kwargs_param, kwargs_fixed_lens, kwargs_fixed_source,
                                 kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo)
        self.param_class.print_setting()

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
                                        kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo, linear_solver=True)
        num_param, list = param_class_linear.num_param()
        assert list[0] == 'theta_E_lens'
        assert num_param == 9

    def test_num_param_linear(self):
        num_param = self.param_class.num_param_linear()
        assert num_param == 4

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

    def test_joint_lens_with_light(self):
        kwargs_model = {'lens_model_list': ['CHAMELEON'], 'lens_light_model_list': ['CHAMELEON']}
        i_light, k_lens = 0, 0
        kwargs_constraints = {'joint_lens_with_light': [[i_light, k_lens, ['w_t', 'w_c', 'center_x', 'center_y', 'e1', 'e2']]]}
        kwargs_lens = [{'theta_E': 10}]
        kwargs_lens_light = [{'amp': 1, 'w_t': 0.5, 'w_c': 0.1, 'center_x': 0, 'center_y': 0.3, 'e1': 0.1, 'e2': -0.2}]
        param = Param(kwargs_model=kwargs_model, kwargs_constraints=kwargs_constraints)
        args = param.setParams(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light)
        kwargs_lens_out, _, kwargs_lens_light_out, _, _ = param.getParams(args)
        assert kwargs_lens_out[0]['w_c'] == kwargs_lens_light[0]['w_c']
        assert kwargs_lens_light_out[0]['w_c'] == kwargs_lens_light[0]['w_c']

        kwargs_model = {'lens_model_list': ['SIS'], 'lens_light_model_list': ['SERSIC']}
        i_light, k_lens = 0, 0
        kwargs_constraints = {'joint_lens_with_light': [[i_light, k_lens, ['center_x',
                                                       'center_y']]]}  # list[[i_point_source, k_source, ['param_name1', 'param_name2', ...]], [
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_lens_light = [{'amp': 1, 'R_sersic': 0.5, 'n_sersic': 2, 'center_x': 1, 'center_y': 1}]
        param = Param(kwargs_model=kwargs_model, kwargs_constraints=kwargs_constraints)
        args = param.setParams(kwargs_lens=kwargs_lens, kwargs_lens_light=kwargs_lens_light)
        kwargs_lens_out, kwargs_source_out, _, kwargs_ps_out, _ = param.getParams(args)
        assert kwargs_lens_out[0]['theta_E'] == kwargs_lens[0]['theta_E']
        assert kwargs_lens_out[0]['center_x'] == kwargs_lens_light[0]['center_x']

    def test_joint_source_with_point_source(self):
        kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'], 'point_source_model_list': ['SOURCE_POSITION']}
        i_source, k_ps = 0, 0
        kwargs_constraints = {'joint_source_with_point_source': [[k_ps, i_source]]} # list[[i_point_source, k_source, ['param_name1', 'param_name2', ...]], [

        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_source = [{'amp': 1, 'n_sersic': 2, 'R_sersic': 0.3, 'center_x': 1, 'center_y': 1}]
        kwargs_ps = [{'ra_source': 0.5, 'dec_source': 0.5}]
        param = Param(kwargs_model=kwargs_model, kwargs_constraints=kwargs_constraints)
        args = param.setParams(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_ps=kwargs_ps)
        kwargs_lens_out, kwargs_source_out, _, kwargs_ps_out, _ = param.getParams(args)
        assert kwargs_lens_out[0]['theta_E'] == kwargs_lens[0]['theta_E']
        assert kwargs_source_out[0]['center_x'] == kwargs_ps[0]['ra_source']

        kwargs_model = {'lens_model_list': ['SIS'], 'source_light_model_list': ['SERSIC'], 'point_source_model_list': ['LENSED_POSITION']}
        i_source, k_ps = 0, 0
        kwargs_constraints = {'joint_source_with_point_source': [[k_ps, i_source]]} # list[[i_point_source, k_source, ['param_name1', 'param_name2', ...]], [

        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        kwargs_source = [{'amp': 1, 'n_sersic': 2, 'R_sersic': 0.3, 'center_x': 1, 'center_y': 1}]
        kwargs_ps = [{'ra_image': [0.5], 'dec_image': [0.5]}]
        param = Param(kwargs_model=kwargs_model, kwargs_constraints=kwargs_constraints)
        args = param.setParams(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_ps=kwargs_ps)
        kwargs_lens_out, kwargs_source_out, _, kwargs_ps_out, _ = param.getParams(args)
        assert kwargs_lens_out[0]['theta_E'] == kwargs_lens[0]['theta_E']
        npt.assert_almost_equal(kwargs_source_out[0]['center_x'], -0.207, decimal=2)


if __name__ == '__main__':
    pytest.main()

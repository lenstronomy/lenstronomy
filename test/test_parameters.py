__author__ = 'sibirrer'

import numpy as np
import pytest
from lenstronomy.Workflow.parameters import Param

class TestParam(object):

    def setup(self):
        kwargs_options = {'lens_type': 'SPEP', 'source_type': 'GAUSSIAN', 'lens_light_type': 'DOUBLE_SERSIC', 'point_source': True
            , 'subgrid_res': 2, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'X2_compare': 'standard', 'X2_type': 'catalogue', 'deltaPix': 0.1
            , 'external_shear': False}
        kwargs_fixed_lens =  {'gamma': 1.9} #for SPEP lens
        kwargs_fixed_source = {'amp': 1, 'sigma_x': 0.1, 'sigma_y': 0.1, 'center_x':0.2, 'center_y':0.2}
        kwargs_fixed_psf = {'sigma': 0.1}
        kwargs_fixed_else = {'point_amp': 1, 'ra_pos': [-1, 1], 'dec_pos': [-1, 1], 'shapelet_beta': 1}
        kwargs_fixed_lens_light = {}
        self.param_class = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_psf, kwargs_fixed_lens_light, kwargs_fixed_else)

    def test_getParams(self):
        kwargs_true_lens =  {'phi_E': 1.,'gamma':1.9,'q':0.8,'phi_G':1.5, 'center_x':0., 'center_y':0.} #for SPEP lens
        kwargs_true_source = {'amp': 1*2*np.pi*0.1**2, 'center_x': 0.2, 'center_y': 0.2, 'sigma_x': 0.1, 'sigma_y': 0.1}
        kwargs_true_psf = {'gaussian': 0.1}
        kwargs_true_lens_light = {'center_x_2': 0.1, 'n_2': 1, 'center_x': -0.06, 'center_y': 0.4, 'phi_G': 4.8,
                                  'q': 0.86, 'R_2': 1.2, 'I0_2': 1.7, 'center_y_2': 0.14, 'n_sersic': 1.7,
                                  'I0_sersic': 11.8, 'R_sersic': 0.697}
        args = self.param_class.setParams(kwargs_true_lens, kwargs_true_source, kwargs_true_psf, kwargs_lens_light=kwargs_true_lens_light)
        lens_dict, source_dict, psf_dict, lens_light_dic, else_dict = self.param_class.getParams(args)
        assert lens_dict['phi_E'] == 1.
        assert lens_dict['gamma'] == 1.9
        assert lens_dict['q'] == 0.8
        assert lens_dict['phi_G'] == 1.5
        assert lens_dict['center_x'] == 0.
        assert lens_dict['center_y'] == 0.
        assert lens_light_dic['center_x'] == -0.06

    def test_param_bound(self):
        low, high = self.param_class.param_bounds()
        assert low[0] == 0.001
        assert high[0] == 10

    def test_num_param(self):
        low, high = self.param_class.param_bounds()
        num_param, list = self.param_class.num_param()
        assert len(low) == num_param
        assert list[0] == 'phi_E_lens'


    def test_get_all_params(self):
        kwargs_true_lens = {'phi_E': 1.,'gamma':1.9,'q':0.8,'phi_G':1.5, 'center_x':0., 'center_y':0.} #for SPEP lens
        kwargs_true_source = {'amp':1*2*np.pi*0.1**2 ,'center_x':0.2, 'center_y':0.2, 'sigma_x': 0.1, 'sigma_y': 0.1}
        kwargs_true_psf = {'gaussian': 0.1}
        kwargs_true_lens_light = {'center_x_2': 0.1, 'n_2': 1, 'center_x': -0.06, 'center_y': 0.4, 'phi_G': 4.8,
                                  'q': 0.86, 'R_2': 1.2, 'I0_2': 1.7, 'center_y_2': 0.14, 'n_sersic': 1.7,
                                  'I0_sersic': 11.8, 'R_sersic': 0.697}
        args = self.param_class.setParams(kwargs_true_lens, kwargs_true_source, kwargs_true_psf, kwargs_lens_light=kwargs_true_lens_light)
        lens_dict, source_dict, psf_dict, lens_light_dic, else_dict = self.param_class.get_all_params(args)
        assert lens_dict['phi_E'] == 1.
        assert lens_dict['gamma'] == 1.9
        assert lens_dict['q'] == 0.8
        assert lens_dict['phi_G'] == 1.5
        assert lens_dict['center_x'] == 0.
        assert lens_dict['center_y'] == 0.
        assert lens_light_dic['center_x'] == -0.06

if __name__ == '__main__':
    pytest.main()
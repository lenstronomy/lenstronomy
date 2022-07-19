__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
import numpy as np
from lenstronomy.Sampling.special_param import SpecialParam


class TestParam(object):

    def setup(self):
        self.param = SpecialParam(Ddt_sampling=True, kwargs_fixed=None, point_source_offset=True, num_images=2,
                                  source_size=True, num_tau0=2, num_z_sampling=3, source_grid_offset=True,
                                  kinematic_sampling=True)
        self.kwargs = {'D_dt': 1988, 'delta_x_image': [0, 0], 'delta_y_image': [0, 0], 'source_size': 0.1,
                       'tau0_list': [0, 1], 'z_sampling': np.array([0.1, 0.5, 2]),
                       'delta_x_source_grid': 0, 'delta_y_source_grid': 0, 'b_ani':0.1, 'incli':0.,'D_d':2000}

    def test_get_setParams(self):
        args = self.param.set_params(self.kwargs)
        kwargs_new, _ = self.param.get_params(args, i=0)
        args_new = self.param.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        param_fixed = SpecialParam(Ddt_sampling=True, kwargs_fixed=self.kwargs, point_source_offset=True, num_images=2,
                                   source_size=True, num_z_sampling=3, num_tau0=2,kinematic_sampling=True)
        kwargs_new, i = param_fixed.get_params(args=[], i=0)
        kwargs_new['D_dt'] = self.kwargs['D_dt']
        kwargs_new['b_ani'] = self.kwargs['b_ani']
        kwargs_new['incli'] = self.kwargs['incli']
        kwargs_new['D_d'] = self.kwargs['D_d']

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 16

    def test_mass_scaling(self):
        kwargs_fixed = {}
        param = SpecialParam(kwargs_fixed=kwargs_fixed, mass_scaling=True, num_scale_factor=3)
        kwargs = {'scale_factor': [0, 1, 2]}
        args = param.set_params(kwargs)
        assert len(args) == 3
        num_param, param_list = param.num_param()
        assert num_param == 3
        kwargs_new, _ = param.get_params(args, i=0)
        assert kwargs_new['scale_factor'][1] == 1
        param = SpecialParam(kwargs_fixed=kwargs, mass_scaling=True, num_scale_factor=3)
        kwargs_in = {'scale_factor': [9, 9, 9]}
        args = param.set_params(kwargs_in)
        assert len(args) == 0
        kwargs_new, _ = param.get_params(args, i=0)
        print(kwargs_new)
        assert kwargs_new['scale_factor'][1] == 1

    def test_delta_images(self):
        param = SpecialParam(num_images=2, point_source_offset=True, kwargs_fixed={},
                             kwargs_lower={'delta_x_image': [-1, -1], 'delta_y_image': [-1, -1]},
                             kwargs_upper={'delta_x_image': [1, 1], 'delta_y_image': [1, 1]})
        kwargs = {'delta_x_image': [0.5, 0.5], 'delta_y_image': [0.5, 0.5]}
        args = param.set_params(kwargs_special=kwargs)
        kwargs_new, _ = param.get_params(args, i=0)
        print(kwargs_new)
        assert kwargs_new['delta_x_image'][0] == kwargs['delta_x_image'][0]

    def test_source_grid_offsets(self):
        param = SpecialParam(kwargs_lower={'delta_x_source_grid': -1, 'delta_y_source_grid': 1},
                             kwargs_upper={'delta_x_source_grid': 1, 'delta_y_source_grid': 1},
                             source_grid_offset=True)
        kwargs = {'delta_x_source_grid': 0.1, 'delta_y_source_grid': 0.1}
        args = param.set_params(kwargs_special=kwargs)
        kwargs_new, _ = param.get_params(args, i=0)
        assert kwargs_new['delta_x_source_grid'] == kwargs['delta_x_source_grid']
        assert kwargs_new['delta_y_source_grid'] == kwargs['delta_y_source_grid']

        kwargs_fixed = {'delta_x_source_grid': 0, 'delta_y_source_grid': 0}
        param = SpecialParam(kwargs_lower={'delta_x_source_grid': -1, 'delta_y_source_grid': 1},
                             kwargs_upper={'delta_x_source_grid': 1, 'delta_y_source_grid': 1},
                             source_grid_offset=True, kwargs_fixed=kwargs_fixed)
        kwargs = {'delta_x_source_grid': 0.1, 'delta_y_source_grid': 0.1}
        args = param.set_params(kwargs_special=kwargs)
        kwargs_new, _ = param.get_params(args, i=0)
        assert kwargs_new['delta_x_source_grid'] == kwargs_fixed['delta_x_source_grid']
        assert kwargs_new['delta_y_source_grid'] == kwargs_fixed['delta_y_source_grid']


if __name__ == '__main__':
    pytest.main()

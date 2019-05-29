__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.cosmo_param import CosmoParam


class TestParam(object):

    def setup(self):
        kwargs_fixed = {}
        self.param = CosmoParam(Ddt_sampling=True, kwargs_fixed=kwargs_fixed, point_source_offset=True, num_images=2,
                                source_size=True)
        self.kwargs = {'D_dt': 1988, 'delta_x_image': [0, 0], 'delta_y_image': [0, 0], 'source_size': 0.1}

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        param_fixed = CosmoParam(Ddt_sampling=True, kwargs_fixed=self.kwargs, point_source_offset=True, num_images=2,
                                source_size=True)
        kwargs_new, i = param_fixed.getParams(args=[], i=0)
        kwargs_new['D_dt'] = self.kwargs['D_dt']

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 6

    def test_mass_scaling(self):
        kwargs_fixed = {}
        param = CosmoParam(kwargs_fixed=kwargs_fixed, mass_scaling=True, num_scale_factor=3)
        kwargs = {'scale_factor': [0, 1, 2]}
        args = param.setParams(kwargs)
        assert len(args) == 3
        num_param, param_list = param.num_param()
        assert num_param == 3
        kwargs_new, _ = param.getParams(args, i=0)
        assert kwargs_new['scale_factor'][1] == 1
        param = CosmoParam(kwargs_fixed=kwargs, mass_scaling=True, num_scale_factor=3)
        kwargs_in = {'scale_factor': [9, 9, 9]}
        args = param.setParams(kwargs_in)
        assert len(args) == 0
        kwargs_new, _ = param.getParams(args, i=0)
        print(kwargs_new)
        assert kwargs_new['scale_factor'][1] == 1

    def test_delta_images(self):
        param = CosmoParam(num_images=2, point_source_offset=True, kwargs_fixed={},
                   kwargs_lower={'delta_x_image': [-1, -1], 'delta_y_image': [-1, -1]},
                   kwargs_upper={'delta_x_image': [1, 1], 'delta_y_image': [1, 1]})
        kwargs = {'delta_x_image': [0.5, 0.5], 'delta_y_image': [0.5, 0.5]}
        args = param.setParams(kwargs_cosmo=kwargs)
        kwargs_new, _ = param.getParams(args, i=0)
        print(kwargs_new)
        assert kwargs_new['delta_x_image'][0] == kwargs['delta_x_image'][0]


if __name__ == '__main__':
    pytest.main()

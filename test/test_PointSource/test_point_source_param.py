__author__ = 'sibirrer'

import pytest
import unittest
import numpy as np
import numpy.testing as npt
from lenstronomy.PointSource.point_source_param import PointSourceParam


class TestParam(object):

    def setup(self):
        kwargs_fixed = [{}, {}, {}]
        num_point_sources_list = [4, 1, 1]
        fixed_magnification_list = [True, False, False]
        point_source_model_list = ['LENSED_POSITION', 'SOURCE_POSITION', 'UNLENSED']
        self.param = PointSourceParam(model_list=point_source_model_list, kwargs_fixed=kwargs_fixed,
                                      num_point_source_list=num_point_sources_list,
                                      fixed_magnification_list=fixed_magnification_list)
        self.kwargs =[{'ra_image': np.array([0, 0, 0, 0]), 'dec_image': np.array([0, 0, 0, 0]),
                       'source_amp': 1},
                      {'ra_source': 1, 'dec_source': 1, 'point_amp': np.array([1.])},
                      {'ra_image': [1], 'dec_image': [1], 'point_amp': np.array([1.])}]

        self.param_linear = PointSourceParam(model_list=point_source_model_list, kwargs_fixed=[{}, {}, {}],
                                      num_point_source_list=num_point_sources_list, linear_solver=False,
                                             fixed_magnification_list=fixed_magnification_list)

    def test_get_setParams(self):
        args = self.param.set_params(self.kwargs)
        kwargs_new, _ = self.param.get_params(args, i=0)
        args_new = self.param.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_linear.set_params(self.kwargs)
        kwargs_new, _ = self.param_linear.get_params(args, i=0)
        args_new = self.param_linear.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 12

        num, list = self.param_linear.num_param()
        assert num == 15

    def test_num_param_linear(self):
        num = self.param.num_param_linear()
        assert num == 3

        num = self.param_linear.num_param_linear()
        assert num == 0

    def test_init(self):
        ps_param = PointSourceParam(model_list=['UNLENSED'], kwargs_fixed=[{}], num_point_source_list=None)
        assert ps_param._num_point_sources_list[0] == 1


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            PointSourceParam(model_list=['BAD'], kwargs_fixed=[{}], kwargs_lower=None, kwargs_upper=[{'bla': 1}])
        with self.assertRaises(ValueError):
            PointSourceParam(model_list=['BAD'], kwargs_fixed=[{}], kwargs_lower=[{'bla': 1}], kwargs_upper=None)


if __name__ == '__main__':
    pytest.main()

__author__ = 'sibirrer'

import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Workflow.else_param import ElseParam


class TestParam(object):

    def setup(self):
        kwargs_options = {'num_images': 4, 'foreground_shear': True, 'mass2light_fixed': True, 'time_delay': True}
        self.param = ElseParam(kwargs_options=kwargs_options, kwargs_fixed={})
        self.kwargs = {'ra_pos': np.array([0, 0, 0, 0]), 'dec_pos': np.array([0 , 0, 0, 0]),
                       'point_amp': np.array([1, 1, 1, 1]), 'gamma1_foreground': 0.1, 'gamma2_foreground': 0.1,
                       'delay_dist': 1000, 'mass2light': 1}
        self.kwargs_sigma = {'pos_sigma': 1, 'point_amp_sigma': 1, 'shear_foreground_sigma': 0.1,
                             'delay_dist_sigma': 100, 'mass2light_sigma': 0.1}
        self.kwargs_mean = dict(self.kwargs.items() + self.kwargs_sigma.items())

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_add2fixed(self):
        kwargs_fixed_used = self.param.add2fix(self.kwargs)
        assert kwargs_fixed_used['delay_dist'] == self.kwargs['delay_dist']

    def test_param_init(self):
        mean, sigma = self.param.param_init(self.kwargs_mean)
        assert mean[0] == 0

    def test_param_bounds(self):
        low, high = self.param.param_bounds()
        assert low[0] == -60

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 16


if __name__ == '__main__':
    pytest.main()

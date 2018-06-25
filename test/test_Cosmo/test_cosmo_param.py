__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.cosmo_param import CosmoParam


class TestParam(object):

    def setup(self):
        cosmo_type = 'D_dt'
        kwargs_fixed = {}
        self.param = CosmoParam(cosmo_type=cosmo_type, kwargs_fixed=kwargs_fixed)
        self.kwargs = {'D_dt': 1988}

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_param_init(self):
        kwargs_mean = {'D_dt': 1000, 'D_dt_sigma': 200}
        mean, sigma = self.param.param_init(kwargs_mean)
        assert mean[0] == 1000

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 1

    def test_mass_scaling(self):
        kwargs_fixed = {}
        param = CosmoParam(kwargs_fixed=kwargs_fixed, mass_scaling=True, num_scale_factor=3)
        kwargs = {'scale_factor': [0, 1, 2]}
        args = param.setParams(kwargs)
        kwargs_mean = kwargs
        kwargs_mean['scale_factor_sigma'] = [1, 1, 1]
        mean, sigma = param.param_init(kwargs_mean=kwargs_mean)
        assert sigma[0] == 1
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


if __name__ == '__main__':
    pytest.main()

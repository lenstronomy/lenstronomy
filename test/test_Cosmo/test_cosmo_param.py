__author__ = 'sibirrer'

import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Cosmo.cosmo_param import CosmoParam


class TestParam(object):

    def setup(self):
        kwargs_cosmo = {'sampling': True, 'D_dt_init': 1000, 'D_dt_sigma': 100, 'D_dt_lower': 0, 'D_dt_upper': 10000}
        self.param = CosmoParam(**kwargs_cosmo)
        self.kwargs = {'D_dt': 1988}

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_param_init(self):
        mean, sigma = self.param.param_init()
        assert mean[0] == 1000

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 1


if __name__ == '__main__':
    pytest.main()

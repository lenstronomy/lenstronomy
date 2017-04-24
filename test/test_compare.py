__author__ = 'sibirrer'


from lenstronomy.MCMC.compare import Compare

import numpy as np
import pytest
#from lenstronomy.unit_manager import UnitManager

class TestCompare(object):

    def setup(self):
        kwargs_options = {'X2_compare': 'simple'}
        self.compare = Compare(kwargs_options)

    def test_compare2D(self):
        data = np.zeros((10,10))
        sim = np.ones((10,10))
        sigma = 1.
        X2 = self.compare.compare2D(sim,data,sigma)
        assert X2 == 100.

    def test_get_marg_const(self):
        M_inv = np.array([[1,0],[0,1]])
        marg_const = self.compare.get_marg_const(M_inv)
        assert marg_const == 0

    def test_get_log_likelihood(self):
        X = np.ones(100)

        logL = self.compare.get_log_likelihood(X, cov_matrix=None)
        assert logL == -50
if __name__ == '__main__':
    pytest.main()
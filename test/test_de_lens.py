__author__ = 'sibirrer'

import numpy as np
from lenstronomy.DeLens.de_lens import DeLens


class TestDeLens(object):

    def setup(self):
        self.deLens = DeLens()

    def test_get_param_WLS(self):
        A = np.array([[1,2,3],[3,2,1]]).T
        C_D_inv = np.array([1,1,1])
        d = np.array([1,2,3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        print result
        assert result[0] == 1.
        assert result[1] == 0.
        assert image[0] == d[0]

        C_D_inv = np.array([0,0,0])
        d = np.array([1,2,3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        print result
        assert result[0] == 0.
        assert result[1] == 0.
        assert image[0] == 0.

    def test_get_covariance_matrix(self):
        d = np.array([1, 2, 3])
        sigma_b = 1
        f = 10.
        result = self.deLens.get_covariance_matrix(d, sigma_b, f)
        assert result[0] == 1.1
        assert result[1] == 1.2

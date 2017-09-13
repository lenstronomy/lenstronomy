__author__ = 'sibirrer'

import numpy as np
import lenstronomy.DeLens.de_lens as DeLens


class TestDeLens(object):

    def setup(self):
        self.deLens = DeLens

    def test_get_param_WLS(self):
        A = np.array([[1,2,3],[3,2,1]]).T
        C_D_inv = np.array([1,1,1])
        d = np.array([1,2,3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        assert result[0] == 1.
        assert result[1] == 0.
        assert image[0] == d[0]

        C_D_inv = np.array([0,0,0])
        d = np.array([1,2,3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        assert result[0] == 0.
        assert result[1] == 0.
        assert image[0] == 0.

    def test_marginalisation_const(self):
        A = np.array([[1,2,3],[3,2,1]]).T
        C_D_inv = np.array([1,1,1])
        d = np.array([1,2,3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        logL_marg = self.deLens.marginalisation_const(cov_error)
        assert logL_marg == -2.2821740957339181

        M_inv = np.array([[1,0],[0,1]])
        marg_const = self.deLens.marginalisation_const(M_inv)
        assert marg_const == 0




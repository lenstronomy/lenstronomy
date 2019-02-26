__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import lenstronomy.ImSim.de_lens as DeLens
import pytest


class TestDeLens(object):

    def setup(self):
        self.deLens = DeLens

    def test_get_param_WLS(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([1, 1, 1])
        d = np.array([1, 2, 3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        assert result[0] == 1.
        assert result[1] == 0.
        assert image[0] == d[0]

        result_new, cov_error_new, image_new = self.deLens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        npt.assert_almost_equal(result_new[0], result[0], decimal=10)
        npt.assert_almost_equal(result_new[1], result[1], decimal=10)
        npt.assert_almost_equal(image_new[0], image[0], decimal=10)

    def test_wls_stability(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([0, 0, 0])
        d = np.array([1, 2, 3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d)
        assert result[0] == 0.
        assert result[1] == 0.
        assert image[0] == 0.

        A = np.array([[1, 2, 1], [1, 2, 1]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        assert result[0] == 0.
        assert result[1] == 0.
        assert image[0] == 0.
        C_D_inv = np.array([1, 1, 1])
        A = np.array([[1., 2., 1. + 10**(-8.9)], [1., 2., 1.]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        result, cov_error, image = self.deLens.get_param_WLS(A, C_D_inv, d, inv_bool=True)
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


if __name__ == '__main__':
    pytest.main()


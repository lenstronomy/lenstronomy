__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
from lenstronomy.ImSim import de_lens
import pytest


class TestDeLens(object):
    def setup_method(self):
        pass

    def test_get_param_WLS(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([1, 1, 1])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d)
        npt.assert_almost_equal(result[0], 1, decimal=8)
        npt.assert_almost_equal(result[1], 0, decimal=8)
        npt.assert_almost_equal(image[0], d[0], decimal=8)

        result_new, cov_error_new, image_new = de_lens.get_param_WLS(
            A, C_D_inv, d, inv_bool=False
        )
        npt.assert_almost_equal(result_new[0], result[0], decimal=10)
        npt.assert_almost_equal(result_new[1], result[1], decimal=10)
        npt.assert_almost_equal(image_new[0], image[0], decimal=10)

    def test_wls_stability(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([0, 0, 0])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d)
        npt.assert_almost_equal(result[0], 0, decimal=8)
        npt.assert_almost_equal(result[1], 0, decimal=8)
        npt.assert_almost_equal(image[0], 0, decimal=8)

        A = np.array([[1, 2, 1], [1, 2, 1]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=False)

        npt.assert_almost_equal(result[0], 0, decimal=8)
        npt.assert_almost_equal(result[1], 0, decimal=8)
        npt.assert_almost_equal(image[0], 0, decimal=8)

        C_D_inv = np.array([1, 1, 1])
        A = np.array([[1.0, 2.0, 1.0 + 10 ** (-8.9)], [1.0, 2.0, 1.0]]).T
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d, inv_bool=True)
        npt.assert_almost_equal(result[0], 0, decimal=8)
        npt.assert_almost_equal(result[1], 0, decimal=8)
        npt.assert_almost_equal(image[0], 0, decimal=8)

    def test_get_param_WLS_interferometry(self):
        M = np.array([[15, 3, 2], [3, 15, 3], [2, 3, 14]])
        b = np.array([4, 2, 1])
        param_amps, M_inv = de_lens.get_param_WLS_interferometry(M, b)
        param_amps_expected = np.array([0.24816754, 0.07993019, 0.01884817])
        M_inv_expected = np.array(
            [
                [0.07015707, -0.01256545, -0.00732984],
                [-0.01256545, 0.07190227, -0.01361257],
                [-0.00732984, -0.01361257, 0.07539267],
            ]
        )
        npt.assert_almost_equal(param_amps, param_amps_expected, decimal=8)
        npt.assert_almost_equal(M_inv, M_inv_expected, decimal=8)

        param_amps1, M_inv1 = de_lens.get_param_WLS_interferometry(M, b, inv_bool=False)
        npt.assert_almost_equal(param_amps1, param_amps_expected, decimal=8)
        assert M_inv1 is None

        # test the unstable case
        M_degenerate = np.array([[0, 0, 0], [0, 15, 3], [0, 3, 14]])
        b_degenerate = np.array([0, 2, 1])
        param_amps_deg, M_inv_deg = de_lens.get_param_WLS_interferometry(
            M_degenerate, b_degenerate
        )
        assert M_inv_deg[0, 0] == 0
        assert M_inv_deg[1, 2] == 0
        assert param_amps_deg[0] == 0
        assert param_amps_deg[2] == 0
        param_amps_deg1, M_inv_deg1 = de_lens.get_param_WLS_interferometry(
            M_degenerate, b_degenerate, inv_bool=False
        )
        assert M_inv_deg1 is None
        assert param_amps_deg1[0] == 0
        assert param_amps_deg1[2] == 0

    def test_marginalisation_const(self):
        A = np.array([[1, 2, 3], [3, 2, 1]]).T
        C_D_inv = np.array([1, 1, 1])
        d = np.array([1, 2, 3])
        result, cov_error, image = de_lens.get_param_WLS(A, C_D_inv, d)
        logL_marg = de_lens.marginalisation_const(cov_error)
        npt.assert_almost_equal(logL_marg, -2.2821740957339181, decimal=8)

        M_inv = np.array([[1, 0], [0, 1]])
        marg_const = de_lens.marginalisation_const(M_inv)
        assert marg_const == 0

    def test_margnialization_new(self):
        M_inv = np.array([[1, -0.5, 1], [-0.5, 3, 0], [1, 0, 2]])
        d_prior = 1000
        m = len(M_inv)
        log_det = de_lens.marginalization_new(M_inv, d_prior=d_prior)
        log_det_old = de_lens.marginalisation_const(M_inv)
        npt.assert_almost_equal(
            log_det,
            log_det_old + m / 2.0 * np.log(np.pi / 2.0) - m * np.log(d_prior),
            decimal=9,
        )

        M_inv = np.array([[1, 1, 1], [0.0, 1.0, 0.0], [1.0, 2.0, 1.0]])
        log_det = de_lens.marginalization_new(M_inv, d_prior=10)
        log_det_old = de_lens.marginalisation_const(M_inv)
        npt.assert_almost_equal(log_det, log_det_old, decimal=9)
        npt.assert_almost_equal(log_det, -(10 ** (15)), decimal=10)

        log_det = de_lens.marginalization_new(M_inv, d_prior=None)
        log_det_old = de_lens.marginalisation_const(M_inv)
        npt.assert_almost_equal(log_det, log_det_old, decimal=9)

    def test_stable_inv(self):
        m = np.diag(np.ones(10) * 2)
        m_inv = de_lens._stable_inv(m)
        m_inv_true = np.diag(np.ones(10) / 2)
        npt.assert_almost_equal(m_inv, m_inv_true)

        m = np.zeros((10, 10))
        m_inv = de_lens._stable_inv(m)
        npt.assert_almost_equal(m_inv, m)

    def test_solve_stable(self):
        m = np.array([[2, 1], [1, 2]])
        r = np.array([2, 1])
        b = de_lens._solve_stable(m, r)
        assert len(b) == 2
        npt.assert_almost_equal(b, [1, 0], decimal=8)

        m = np.array([[0, 0], [0, 0]])
        r = np.array([1, 1])
        b_none = de_lens._solve_stable(m, r)
        npt.assert_almost_equal(b_none, [0, 0], decimal=8)
        assert np.shape(b_none) == np.shape(b)


if __name__ == "__main__":
    pytest.main()

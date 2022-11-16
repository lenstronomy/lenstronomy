__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.convergence import Convergence

import numpy as np
import numpy.testing as npt
import pytest


class TestConvergence(object):
    """
    tests the Gaussian methods
    """
    def setup_method(self):
        self.profile = Convergence()
        self.kwargs_lens = {'kappa': 0.1}

    def test_function(self):
        x = np.array([1])
        y = np.array([0])
        values = self.profile.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], self.kwargs_lens['kappa']/2, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.profile.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0, decimal=5)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0],  0.25, decimal=5)
        npt.assert_almost_equal(values[1], 0.5, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.profile.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0], 0.1, decimal=5)
        npt.assert_almost_equal(f_y[0], 0.2, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.profile.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.1, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.2, decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0.1, decimal=5)
        npt.assert_almost_equal(f_yy, 0.1, decimal=5)
        npt.assert_almost_equal(f_xy, 0, decimal=5)
        npt.assert_almost_equal(f_yx, 0, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.profile.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.1, decimal=5)
        npt.assert_almost_equal(values[3], 0.1, decimal=5)
        npt.assert_almost_equal(values[1], 0, decimal=5)


if __name__ == '__main__':
    pytest.main()

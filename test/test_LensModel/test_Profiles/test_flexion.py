__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.flexion import Flexion

import numpy as np
import numpy.testing as npt
import pytest

class TestExternalShear(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.flex = Flexion()

        g1, g2, g3, g4 = 0.01, 0.01, 0.01, 0.01
        self.kwargs_lens = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.025, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0, decimal=5)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.flex.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0],  0.025, decimal=5)
        npt.assert_almost_equal(values[1], 0.066666666666666666, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.flex.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0], 0.018333333333333333, decimal=5)
        npt.assert_almost_equal(f_y[0], 0.028333333333333335, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.flex.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.018333333333333333, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.028333333333333335, decimal=5)

    def test_hessian(self):
        x = np.array(1)
        y = np.array(2)
        f_xx, f_yy, f_xy = self.flex.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0.016666666666666666, decimal=5)
        npt.assert_almost_equal(f_yy, 0.023333333333333334, decimal=5)
        npt.assert_almost_equal(f_xy, 0.01, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.flex.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.016666666666666666, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.023333333333333334, decimal=5)
        npt.assert_almost_equal(values[2][0], 0.01, decimal=5)


if __name__ == '__main__':
    pytest.main()
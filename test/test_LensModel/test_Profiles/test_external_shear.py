__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.external_shear import ExternalShear

import numpy as np
import numpy.testing as npt
import pytest

class TestExternalShear(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.extShear = ExternalShear()

        e1, e2 = 0.1, 0.1
        self.kwargs_lens = {'e1': e1, 'e2': e2}

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        values = self.extShear.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.05, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.extShear.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0, decimal=5)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.extShear.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0],  0.35, decimal=5)
        npt.assert_almost_equal(values[1], 0.7, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.extShear.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0], 0.3, decimal=5)
        npt.assert_almost_equal(f_y[0], -0.1, decimal=5)

        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        values = self.extShear.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0][0], 0.3, decimal=5)
        npt.assert_almost_equal(values[1][0], -0.1, decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        f_xx, f_yy, f_xy = self.extShear.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0.1, decimal=5)
        npt.assert_almost_equal(f_yy, -0.1, decimal=5)
        npt.assert_almost_equal(f_xy, 0.1, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.extShear.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0.1, decimal=5)
        npt.assert_almost_equal(values[1], -0.1, decimal=5)
        npt.assert_almost_equal(values[2], 0.1, decimal=5)


if __name__ == '__main__':
    pytest.main()
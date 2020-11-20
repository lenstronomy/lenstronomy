__author__ = 'sibirrer'

import unittest
from lenstronomy.LensModel.Profiles.hessian import Hessian
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest


class TestHessian(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.hessian = Hessian()

        self.f_xx, self.f_yy, self.f_xy, self.f_yx = 0.1, 0.1, -0.1, -0.1
        self.kwargs_lens = {'f_xx': self.f_xx, 'f_yy': self.f_yy, 'f_xy': self.f_xy, 'f_yx': self.f_yx}

    def test_function(self):
        x = 1
        y = 2
        values = self.hessian.function(x, y, **self.kwargs_lens)
        f_true = 1/2. * (self.f_xx * x**2 + (self.f_xy + self.f_yx) * x * y + self.f_yy * y**2)
        npt.assert_almost_equal(values, f_true, decimal=5)
        x = np.array([0])
        y = np.array([0])
        values = self.hessian.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(values[0], 0, decimal=5)

    def test_derivatives(self):
        x = 1
        y = 2
        f_x, f_y = self.hessian.derivatives(x, y, **self.kwargs_lens)
        fx_true = self.f_xx * x + self.f_xy * y
        fy_true = self.f_yx * x + self.f_yy * y
        npt.assert_almost_equal(f_x, fx_true, decimal=5)
        npt.assert_almost_equal(f_y, fy_true, decimal=5)

    def test_hessian(self):
        x = 1
        y = 2

        f_xx, f_yy, f_xy = self.hessian.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, self.f_xx, decimal=5)
        npt.assert_almost_equal(f_yy, self.f_yy, decimal=5)
        npt.assert_almost_equal(f_xy, self.f_xy, decimal=5)

        lensModel = LensModel(['HESSIAN'])
        f_xy_true, f_yx_true = 0.3, 0.2
        kwargs_lens = {'f_xx': self.f_xx, 'f_yy': self.f_yy, 'f_xy': f_xy_true, 'f_yx': f_yx_true}
        f_xx, f_xy, f_yx, f_yy = lensModel.hessian(x, y, [kwargs_lens], diff=0.001)
        npt.assert_almost_equal(f_xx, self.f_xx, decimal=9)
        npt.assert_almost_equal(f_yy, self.f_yy, decimal=9)
        npt.assert_almost_equal(f_xy, f_xy_true, decimal=9)
        npt.assert_almost_equal(f_yx, f_yx_true, decimal=9)


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            hessian = Hessian()
            hessian.hessian(0, 0, f_xx=0, f_yy=0, f_xy=-1, f_yx=1)


if __name__ == '__main__':
    pytest.main()

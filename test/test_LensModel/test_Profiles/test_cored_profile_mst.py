__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.cored_density_mst import CoredDensityMST

import numpy as np
import numpy.testing as npt
import pytest
import unittest


class TestMassSheet(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.profile1 = CoredDensityMST(profile_type='CORED_DENSITY')
        self.profile2 = CoredDensityMST(profile_type='CORED_DENSITY_2')
        self.profile3 = CoredDensityMST(profile_type='CORED_DENSITY_EXP')
        self.profile4 = CoredDensityMST(profile_type='CORED_DENSITY_ULDM')
        self.kwargs_lens = {'lambda_approx': 0.9, 'r_core': 100, 'center_x': 0, 'center_y': 0}

    def test_function(self):
        x = np.array([0.01, 1])
        y = np.array([0, 0])
        f_ = self.profile1.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_[0] - f_[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_ = self.profile2.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_[0] - f_[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_ = self.profile3.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_[0] - f_[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_ = self.profile4.function(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_[0] - f_[1], 0, decimal=3)  # test to demand that the profile is (almost) zero

    def test_derivatives(self):
        x = np.array([0.01, 1])
        y = np.array([0, 0])
        f_x, f_y = self.profile1.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0] - f_x[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_x, f_y = self.profile2.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0] - f_x[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_x, f_y = self.profile3.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0] - f_x[1], 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_x, f_y = self.profile4.derivatives(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_x[0] - f_x[1], 0, decimal=3)  # test to demand that the profile is (almost) zero

    def test_hessian(self):
        x = np.array([0.01, 1])
        y = np.array([0, 0])
        f_xx, f_yy, f_xy = self.profile1.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_xx, f_yy, f_xy = self.profile2.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_xx, f_yy, f_xy = self.profile3.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0, decimal=3)  # test to demand that the profile is (almost) zero
        f_xx, f_yy, f_xy = self.profile4.hessian(x, y, **self.kwargs_lens)
        npt.assert_almost_equal(f_xx, 0, decimal=3)  # test to demand that the profile is (almost) zero


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            CoredDensityMST(profile_type='WRONG_PROFILE')


if __name__ == '__main__':
    pytest.main()

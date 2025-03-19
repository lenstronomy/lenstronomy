__author__ = "nataliehogg"

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS


class TestLOS(object):
    """Tests the LOS profile."""

    def setup_method(self):
        self.LOS = LOS()

    def test_distort_vector(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        x = 1
        y = 1

        x_distorted, y_distorted = self.LOS.distort_vector(
            x, y, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(x_distorted, 0.8, decimal=9)
        npt.assert_almost_equal(y_distorted, 0.8, decimal=9)

    def test_left_multiply(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        fxx = 1
        fxy = 1
        fyx = 1
        fyy = 1

        f_xx, f_xy, f_yx, f_yy = self.LOS.left_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(f_xx, 0.8)
        npt.assert_almost_equal(f_xy, 0.8)
        npt.assert_almost_equal(f_yx, 0.8)
        npt.assert_almost_equal(f_yy, 0.8)

    def test_right_multiply(self):
        kappa = 0.1
        gamma1 = 0.2
        gamma2 = 0.1
        omega = 0.2
        fxx = 1
        fxy = 1
        fyx = 1
        fyy = 1

        f_xx, f_xy, f_yx, f_yy = self.LOS.right_multiply(
            fxx, fxy, fyx, fyy, kappa, gamma1, gamma2, omega
        )

        npt.assert_almost_equal(f_xx, 0.4)
        npt.assert_almost_equal(f_xy, 1.2)
        npt.assert_almost_equal(f_yx, 0.4)
        npt.assert_almost_equal(f_yy, 1.2)

    def test_set_static(self):

        p = self.LOS.set_static()

    def test_set_dynamic(self):

        d = self.LOS.set_dynamic()


if __name__ == "__main__":
    pytest.main("-k TestLOS")

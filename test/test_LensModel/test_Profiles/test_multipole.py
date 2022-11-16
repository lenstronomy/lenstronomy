__author__ = 'lynevdv'


from lenstronomy.LensModel.Profiles.multipole import Multipole

import numpy as np
import pytest
import numpy.testing as npt


class TestMultipole(object):
    """
    tests the Gaussian methods
    """
    def setup_method(self):
        self.Multipole = Multipole()

    def test_function(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25*np.pi/180.
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values, 0.006684307, decimal=6)
        x = np.array([0])
        y = np.array([0])
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        assert values[0] == 0

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.Multipole.function(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values[0], -0.007409114, decimal=6)
        npt.assert_almost_equal(values[1], -0.009453038, decimal=6)
        npt.assert_almost_equal(values[2], -0.009910505, decimal=6)

    def test_derivatives(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x, -0.003939644, decimal=6)
        npt.assert_almost_equal(f_y, 0.005311976, decimal=6)

        x = np.array([1])
        y = np.array([2])
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x[0], -0.003939644, decimal=6)
        npt.assert_almost_equal(f_y[0], 0.005311976, decimal=6)

        x = np.array([2, 3, 1])
        y = np.array([1, 1, 4])
        f_x, f_y = self.Multipole.derivatives(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_x[0], -0.003613858, decimal=6)
        npt.assert_almost_equal(f_x[1], -0.000970385, decimal=6)
        npt.assert_almost_equal(f_x[2], 0.005970704, decimal=6)
        npt.assert_almost_equal(f_y[0], -0.000181398, decimal=6)
        npt.assert_almost_equal(f_y[1], -0.006541883, decimal=6)
        npt.assert_almost_equal(f_y[2], 0.001649720, decimal=6)

    def test_hessian(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.
        f_xx, f_xy, f_yx, f_yy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx, -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy, -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy, 0.008021169, decimal=6)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)
        x = np.array([1])
        y = np.array([2])
        f_xx, f_xy, f_yx, f_yy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx[0], -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy[0], -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy[0], 0.008021169, decimal=6)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values[0][0], -0.016042338, decimal=6)
        npt.assert_almost_equal(values[3][0], -0.004010584, decimal=6)
        npt.assert_almost_equal(values[1][0], 0.008021169, decimal=6)
        npt.assert_almost_equal(values[0][1], 0.001417956, decimal=6)
        npt.assert_almost_equal(values[3][1], 0.012761602, decimal=6)
        npt.assert_almost_equal(values[1][1], -0.004253867, decimal=6)


if __name__ == '__main__':
   pytest.main()

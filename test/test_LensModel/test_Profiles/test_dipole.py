__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.dipole import Dipole, Dipole_util

import numpy as np
import numpy.testing as npt
import pytest

class TestDipole(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.dipole = Dipole()
        self.dipole_util = Dipole_util()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        center1_x = 0
        center1_y = 0
        center2_x = 1
        center2_y = -1
        c = 1.
        Fm = 0.5
        com_x, com_y = self.dipole_util.com(center1_x, center1_y, center2_x, center2_y, Fm)
        phi_dipole = self.dipole_util.angle(center1_x, center1_y, center2_x, center2_y)
        values = self.dipole.function(x, y, com_x, com_y, phi_dipole, c)
        #npt.assert_almost_equal(values[0], 0, decimal=5)
        x = np.array([0])
        y = np.array([0])

        values = self.dipole.function(x, y, com_x, com_y, phi_dipole, c)
        #npt.assert_almost_equal(values[0], 0, decimal=5)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.dipole.function(x, y, com_x, com_y, phi_dipole, c)
        #npt.assert_almost_equal(values[0], 0, decimal=5)
        #npt.assert_almost_equal(values[1], 0, decimal=5)
        #npt.assert_almost_equal(values[2], 0, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        center1_x = 0
        center1_y = 0
        center2_x = 1
        center2_y = -1
        c = 1.
        Fm = 0.5
        com_x, com_y = self.dipole_util.com(center1_x, center1_y, center2_x, center2_y, Fm)
        phi_dipole = self.dipole_util.angle(center1_x, center1_y, center2_x, center2_y)
        f_x, f_y = self.dipole.derivatives(x, y, com_x, com_y, phi_dipole, c)
        npt.assert_almost_equal(f_x[0], -0.43412157106222954, decimal=5)
        npt.assert_almost_equal(f_y[0], 0.43412157106222948, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.dipole.derivatives(x, y, com_x, com_y, phi_dipole, c)
        npt.assert_almost_equal(values[0][0], -0.43412157106222954, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.43412157106222948, decimal=5)
        npt.assert_almost_equal(values[0][1], 0.11624763874381937, decimal=5)
        npt.assert_almost_equal(values[1][1], -0.11624763874381935, decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        center1_x = 0
        center1_y = 0
        center2_x = 1
        center2_y = -1
        c = 1.
        Fm = 0.5
        com_x, com_y = self.dipole_util.com(center1_x, center1_y, center2_x, center2_y, Fm)
        phi_dipole = self.dipole_util.angle(center1_x, center1_y, center2_x, center2_y)
        f_xx, f_yy,f_xy = self.dipole.hessian(x, y, com_x, com_y, phi_dipole, c)
        npt.assert_almost_equal(f_xx[0], 0.29625219299960942, decimal=5)
        npt.assert_almost_equal(f_yy[0], -0.064402650652089, decimal=5)
        npt.assert_almost_equal(f_xy[0], -0.1159247711737602, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.dipole.hessian(x, y, com_x, com_y, phi_dipole, c)
        npt.assert_almost_equal(values[0][0], 0.29625219299960942, decimal=5)
        npt.assert_almost_equal(values[1][0], -0.064402650652089, decimal=5)
        npt.assert_almost_equal(values[2][0], -0.1159247711737602, decimal=5)
        npt.assert_almost_equal(values[0][1], 0.11310581066966192, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.22621162133932399, decimal=5)
        npt.assert_almost_equal(values[2][1], -0.16965871600449295, decimal=5)

    def test_mass_ratio(self):
        ratio = self.dipole_util.mass_ratio(theta_E=1., theta_E_sub=0.1)
        assert ratio == 100


if __name__ == '__main__':
    pytest.main()
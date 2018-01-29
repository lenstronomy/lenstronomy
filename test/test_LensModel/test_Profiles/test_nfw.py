__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE

import numpy as np
import numpy.testing as npt
import pytest

class TestNFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        values = self.nfw.function(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(values[0], 2.4764530888727556, decimal=5)
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        values = self.nfw.function(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(values[0], 0, decimal=4)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.nfw.function(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(values[0], 2.4764530888727556, decimal=5)
        npt.assert_almost_equal(values[1], 3.5400250357511416, decimal=5)
        npt.assert_almost_equal(values[2], 4.5623722261790647, decimal=5)


    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_x, f_y = self.nfw.derivatives(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(f_x[0], 0.53211690764331998, decimal=5)
        npt.assert_almost_equal(f_y[0], 1.06423381528664, decimal=5)
        x = np.array([0])
        y = np.array([0])
        theta_Rs = 0
        f_x, f_y = self.nfw.derivatives(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(f_x[0], 0, decimal=5)
        npt.assert_almost_equal(f_y[0], 0, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        values = self.nfw.derivatives(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(values[0][0], 0.53211690764331998, decimal=5)
        npt.assert_almost_equal(values[1][0], 1.06423381528664, decimal=5)
        npt.assert_almost_equal(values[0][1], 1.0493927480837946, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.34979758269459821, decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        rho0 = 1
        theta_Rs = self.nfw._rho02alpha(rho0, Rs)
        f_xx, f_yy,f_xy = self.nfw.hessian(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(f_xx[0], 0.40855527280658294, decimal=5)
        npt.assert_almost_equal(f_yy[0], 0.037870368296371637, decimal=5)
        npt.assert_almost_equal(f_xy[0], -0.2471232696734742, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.nfw.hessian(x, y, Rs, theta_Rs)
        npt.assert_almost_equal(values[0][0], 0.40855527280658294, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.037870368296371637, decimal=5)
        npt.assert_almost_equal(values[2][0], -0.2471232696734742, decimal=5)
        npt.assert_almost_equal(values[0][1], -0.046377502475445781, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.30577812878681554, decimal=5)
        npt.assert_almost_equal(values[2][1], -0.13205836172334798, decimal=5)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        theta_Rs = 1
        m_3d = self.nfw.mass_3d_lens(R, Rs, theta_Rs)
        npt.assert_almost_equal(m_3d, 1.1573795105019022, decimal=8)


class TestMassAngleConversion(object):
    """
    test angular to mass unit conversions
    """
    def setup(self):
        self.nfw = NFW()
        self.nfw_ellipse = NFW_ELLIPSE()

    def test_angle(self):
        x, y = 1, 0
        alpha1, alpha2 = self.nfw.derivatives(x, y, theta_Rs=1., Rs=1.)
        assert alpha1 == 1.

    def test_convertAngle2rho(self):
        rho0 = self.nfw._alpha2rho0(theta_Rs=1., Rs=1.)
        assert rho0 == 0.81472283831773229

    def test_convertrho02angle(self):
        theta_Rs_in = 1.5
        Rs = 1.5
        rho0 = self.nfw._alpha2rho0(theta_Rs=theta_Rs_in, Rs=Rs)
        theta_Rs_out = self.nfw._rho02alpha(rho0, Rs)
        assert theta_Rs_in == theta_Rs_out


if __name__ == '__main__':
    pytest.main()
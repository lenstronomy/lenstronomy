__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe

import numpy as np
import numpy.testing as npt
import pytest


class TestP_JAFFW(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.profile = PJaffe()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        values = self.profile.function(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values[0], 0.87301557036070054, decimal=8)
        x = np.array([0])
        y = np.array([0])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        values = self.profile.function(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values[0], 0.20267440905756931, decimal=8)

        x = np.array([2, 3, 4])
        y = np.array([1, 1, 1])
        values = self.profile.function( x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values[0], 0.87301557036070054, decimal=8)
        npt.assert_almost_equal(values[1], 1.0842781309377669, decimal=8)
        npt.assert_almost_equal(values[2], 1.2588604178849985, decimal=8)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(f_x[0], 0.11542369603751264, decimal=8)
        npt.assert_almost_equal(f_y[0], 0.23084739207502528, decimal=8)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Ra, Rs)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.profile.derivatives(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values[0][0], 0.11542369603751264, decimal=8)
        npt.assert_almost_equal(values[1][0], 0.23084739207502528, decimal=8)
        npt.assert_almost_equal(values[0][1], 0.19172866612512479, decimal=8)
        npt.assert_almost_equal(values[1][1], 0.063909555375041588, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        sigma0 = 1.
        Ra, Rs = 0.5, 0.8
        f_xx, f_xy, f_yx, f_yy = self.profile.hessian(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(f_xx[0], 0.077446121589827679, decimal=8)
        npt.assert_almost_equal(f_yy[0], -0.036486601753227141, decimal=8)
        npt.assert_almost_equal(f_xy[0], -0.075955148895369876, decimal=8)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.profile.hessian(x, y, sigma0, Ra, Rs)
        npt.assert_almost_equal(values[0][0], 0.077446121589827679, decimal=8)
        npt.assert_almost_equal(values[3][0], -0.036486601753227141, decimal=8)
        npt.assert_almost_equal(values[1][0], values[2][0], decimal=8)

    def test_mass_tot(self):
        rho0 = 1.
        Ra, Rs = 0.5, 0.8
        values = self.profile.mass_tot(rho0, Ra, Rs)
        npt.assert_almost_equal(values, 2.429441083345073, decimal=10)

    def test_mass_3d_lens(self):
        mass = self.profile.mass_3d_lens(r=1, sigma0=1, Ra=0.5, Rs=0.8)
        npt.assert_almost_equal(mass, 0.87077306005349242, decimal=8)

    def test_grav_pot(self):
        x = 1
        y = 2
        rho0 = 1.
        r = np.sqrt(x**2 + y**2)
        Ra, Rs = 0.5, 0.8
        grav_pot = self.profile.grav_pot(r, rho0, Ra, Rs)
        npt.assert_almost_equal(grav_pot, 0.89106542283974155, decimal=10)


if __name__ == '__main__':
    pytest.main()

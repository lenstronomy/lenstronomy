__author__ = 'sibirrer'

import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.LensModel.Profiles.hernquist import Hernquist
from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse
import lenstronomy.Util.param_util as param_util


class TestHernquist(object):

    def setup(self):
        self.profile = Hernquist()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function(x, y, sigma0, Rs)
        npt.assert_almost_equal(values[0], 0.66514613455415028, decimal=8)
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function( x, y, sigma0, Rs)
        npt.assert_almost_equal(values[0], 0, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.profile.function( x, y, sigma0, Rs)
        npt.assert_almost_equal(values[0], 0.66514613455415028, decimal=8)
        npt.assert_almost_equal(values[1], 0.87449395673649566, decimal=8)
        npt.assert_almost_equal(values[2], 1.0549139073851708, decimal=8)

    def test_derivatives(self):
        x = 1
        y = 2
        Rs = 1.
        sigma0 = 0.5
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs)
        npt.assert_almost_equal(f_x, 0.11160641027573866, decimal=8)
        npt.assert_almost_equal(f_y, 0.22321282055147731, decimal=8)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs)
        npt.assert_almost_equal(f_x, 0, decimal=8)
        npt.assert_almost_equal(f_y, 0, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        f_xx, f_yy,f_xy = self.profile.hessian(x, y, sigma0, Rs)
        npt.assert_almost_equal(f_xx[0], 0.0779016004481825, decimal=6)
        npt.assert_almost_equal(f_yy[0], -0.023212809452388683, decimal=6)
        npt.assert_almost_equal(f_xy[0], -0.0674096084507525, decimal=6)

    def test_mass_tot(self):
        rho0 = 1
        Rs = 3
        m_tot = self.profile.mass_tot(rho0, Rs)
        npt.assert_almost_equal(m_tot, 169.64600329384882, decimal=6)

    def test_grav_pot(self):
        x, y = 1, 0
        rho0 = 1
        Rs = 3
        grav_pot = self.profile.grav_pot(x, y, rho0, Rs, center_x=0, center_y=0)
        npt.assert_almost_equal(grav_pot, 42.411500823462205, decimal=8)


class TestHernquistEllipse(object):

    def setup(self):
        self.profile = Hernquist_Ellipse()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.profile.function(x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(values[0], 0.6451374041763912, decimal=8)
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function(x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(values[0], 0, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.profile.function( x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(values[0], 0.60477384241056542, decimal=8)
        npt.assert_almost_equal(values[1], 0.80854098526603968, decimal=8)
        npt.assert_almost_equal(values[2], 0.98780932325084092, decimal=8)

    def test_derivatives(self):
        x = 1
        y = 2
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(f_x, 0.065446024625706908, decimal=8)
        npt.assert_almost_equal(f_y, 0.24132860718623173, decimal=8)
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives(x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(f_x, 0, decimal=8)
        npt.assert_almost_equal(f_y, 0, decimal=8)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_yy,f_xy = self.profile.hessian(x, y, sigma0, Rs, e1, e2)
        npt.assert_almost_equal(f_xx[0], 0.09340916928834986, decimal=6)
        npt.assert_almost_equal(f_yy[0], -0.02853883795950196, decimal=6)
        npt.assert_almost_equal(f_xy[0], -0.06298489507727822, decimal=6)


if __name__ == '__main__':
    pytest.main()

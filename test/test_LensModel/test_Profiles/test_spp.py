__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.spep import SPEP
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.sis import SIS

import numpy as np
import numpy.testing as npt
import pytest

class TestSPEP(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SPEP = SPEP()
        self.SPP = SPP()
        self.SIS = SIS()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]
        x = np.array([0])
        y = np.array([0])
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values_spep = self.SPEP.function(x, y, E, gamma,q,phi_G)
        values_spp = self.SPP.function(x, y, E, gamma)
        assert values_spep[0] == values_spp[0]
        assert values_spep[1] == values_spp[1]
        assert values_spep[2] == values_spp[2]

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]
        x = np.array([0])
        y = np.array([0])
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        f_x_spep, f_y_spep = self.SPEP.derivatives(x, y, E, gamma,q,phi_G)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, E, gamma)
        assert f_x_spep[0] == f_x_spp[0]
        assert f_y_spep[0] == f_y_spp[0]
        assert f_x_spep[1] == f_x_spp[1]
        assert f_y_spep[1] == f_y_spp[1]
        assert f_x_spep[2] == f_x_spp[2]
        assert f_y_spep[2] == f_y_spp[2]

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        gamma = 1.9
        q = 1.
        phi_G = 0.
        E = phi_E / (((3-gamma)/2.)**(1./(1-gamma))*np.sqrt(q))
        f_xx, f_yy,f_xy = self.SPEP.hessian( x, y, E,gamma,q,phi_G)
        f_xx_spep, f_yy_spep, f_xy_spep = self.SPEP.hessian(x, y, E, gamma,q,phi_G)
        f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.hessian(x, y, E, gamma)
        assert f_xx_spep[0] == f_xx_spp[0]
        assert f_yy_spep[0] == f_yy_spp[0]
        assert f_xy_spep[0] == f_xy_spp[0]
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        f_xx_spep, f_yy_spep, f_xy_spep = self.SPEP.hessian(x, y, E, gamma,q,phi_G)
        f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.hessian(x, y, E, gamma)
        assert f_xx_spep[0] == f_xx_spp[0]
        assert f_yy_spep[0] == f_yy_spp[0]
        assert f_xy_spep[0] == f_xy_spp[0]
        assert f_xx_spep[1] == f_xx_spp[1]
        assert f_yy_spep[1] == f_yy_spp[1]
        assert f_xy_spep[1] == f_xy_spp[1]
        assert f_xx_spep[2] == f_xx_spp[2]
        assert f_yy_spep[2] == f_yy_spp[2]
        assert f_xy_spep[2] == f_xy_spp[2]

    def test_compare_sis(self):
        x = np.array([1])
        y = np.array([2])
        theta_E = 1.
        gamma = 2.
        f_sis = self.SIS.function( x, y, theta_E)
        f_spp = self.SPP.function(x, y, theta_E, gamma)
        f_x_sis, f_y_sis = self.SIS.derivatives( x, y, theta_E)
        f_x_spp, f_y_spp = self.SPP.derivatives(x, y, theta_E, gamma)
        f_xx_sis, f_yy_sis, f_xy_sis = self.SIS.hessian( x, y, theta_E)
        f_xx_spp, f_yy_spp, f_xy_spp = self.SPP.hessian(x, y, theta_E, gamma)
        npt.assert_almost_equal(f_sis[0],f_spp[0], decimal=7)
        npt.assert_almost_equal(f_x_sis[0], f_x_spp[0], decimal=7)
        npt.assert_almost_equal(f_y_sis[0], f_y_spp[0], decimal=7)
        npt.assert_almost_equal(f_xx_sis[0], f_xx_spp[0], decimal=7)
        npt.assert_almost_equal(f_yy_sis[0], f_yy_spp[0], decimal=7)
        npt.assert_almost_equal(f_xy_sis[0], f_xy_spp[0], decimal=7)

    def test_unit_conversion(self):
        theta_E = 2.
        gamma = 2.2
        rho0 = self.SPP.theta2rho(theta_E, gamma)
        theta_E_out = self.SPP.rho2theta(rho0, gamma)
        assert theta_E == theta_E_out

    def test_mass_2d_lens(self):
        r = 1
        theta_E = 1
        gamma = 2
        m_2d = self.SPP.mass_2d_lens(r, theta_E, gamma)
        npt.assert_almost_equal(m_2d, 3.1415926535897931, decimal=8)

    def test_grav_pot(self):
        x, y = 1, 0
        rho0 = 1
        gamma = 2
        grav_pot = self.SPP.grav_pot(x, y, rho0, gamma, center_x=0, center_y=0)
        npt.assert_almost_equal(grav_pot, 12.566370614359172, decimal=8)


if __name__ == '__main__':
   pytest.main()
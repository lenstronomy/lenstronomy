__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.nfw_param import NFWParam

class TestLensCosmo(object):
    """
    tests the UnitManager class routines
    """
    def setup(self):
        z_L = 0.8
        z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.nfwParam = NFWParam()

    def test_rho0_c(self):
        c = 4
        rho0 = self.nfwParam.rho0_c(c)
        c_out = self.nfwParam.c_rho0(rho0)
        npt.assert_almost_equal(c_out, c, decimal=3)

    def test_rhoc_z(self):
        z = 1
        rho0_z = self.nfwParam.rhoc_z(z=z)
        npt.assert_almost_equal(self.nfwParam.rhoc *(1+z)**3, rho0_z)

    def test_M200(self):
        M200 = self.nfwParam.M200(Rs=1, rho0=1, c=1)
        npt.assert_almost_equal(M200, 2.4271590540348216, decimal=5)

    def test_profileMain(self):
        M = 10**(13.5)
        z = 0.5
        r200, rho0, c, Rs = self.nfwParam.profileMain(M, z)

        c_ = self.nfwParam.c_M_z(M, z)
        r200_ = self.nfwParam.r200_M(M)
        rho0_ = self.nfwParam.rho0_c(c)
        Rs_ = r200_ / c_
        npt.assert_almost_equal(c_, c, decimal=5)
        npt.assert_almost_equal(r200_, r200, decimal=5)
        npt.assert_almost_equal(rho0_, rho0, decimal=5)
        npt.assert_almost_equal(Rs_, Rs, decimal=5)


if __name__ == '__main__':
    pytest.main()

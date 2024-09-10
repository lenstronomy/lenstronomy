__author__ = "sibirrer", "gilmanda", "ajshajib"

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.nfw_param import NFWParam
from lenstronomy.Cosmo.gnfw_param import GNFWParam
from astropy.cosmology import FlatLambdaCDM


class TestGNFWParam(object):
    """Tests the UnitManager class routines."""

    def setup_method(self):
        self.gnfwParam = GNFWParam(cosmo=None)
        self.nfwParam = NFWParam(cosmo=None)
        self.z = 0.5  # needed fixed redshift for the inversion function

    def test_rho0_c(self):
        c_list = [0.1, 1, 4, 10, 20]
        for c in c_list:
            rho0 = self.gnfwParam.rho0_c(c, z=self.z, gamma_in=0.8)
            c_out = self.gnfwParam.c_rho0(rho0, z=self.z, gamma_in=0.8)
            print(c, "c")
            npt.assert_almost_equal(c_out, c, decimal=3)

            nfw_rho0 = self.nfwParam.rho0_c(c, z=self.z)
            gnfw_rho0 = self.gnfwParam.rho0_c(c, z=self.z, gamma_in=1)
            npt.assert_almost_equal(nfw_rho0 / gnfw_rho0, 1.0, decimal=10)

    def test_M_r200(self):
        r200 = 200
        M200 = self.gnfwParam.M_r200(r200, z=self.z)
        M200_nfw = self.nfwParam.M_r200(r200, z=self.z)
        npt.assert_almost_equal(M200 / M200_nfw, 1, decimal=10)

    def test_rhoc_z(self):
        z = 0
        rho0_z = self.gnfwParam.rhoc_z(z=z)
        npt.assert_almost_equal(self.gnfwParam.rhoc * (1 + z) ** 3, rho0_z)

    def test_M200(self):
        M200 = self.gnfwParam.M200(rs=1, rho0=1, c=1, gamma_in=1.0)
        M200_nfw = self.nfwParam.M200(rs=1, rho0=1, c=1)
        npt.assert_almost_equal(M200 / M200_nfw, 1.0, decimal=10)

    def test_profileMain(self):
        M = 10**13.5
        z = 0.5
        gamma_in = 0.8
        r200, rho0, c, Rs = self.gnfwParam.gnfw_Mz(M, z, gamma_in)

        c_ = self.gnfwParam.c_M_z(M, z)
        r200_ = self.gnfwParam.r200_M(M, z)
        rho0_ = self.gnfwParam.rho0_c(c, z, gamma_in)
        Rs_ = r200_ / c_
        npt.assert_almost_equal(c_, c, decimal=5)
        npt.assert_almost_equal(r200_, r200, decimal=5)
        npt.assert_almost_equal(rho0_, rho0, decimal=5)
        npt.assert_almost_equal(Rs_, Rs, decimal=5)


if __name__ == "__main__":
    pytest.main()

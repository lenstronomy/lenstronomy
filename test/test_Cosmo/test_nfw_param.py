__author__ = 'sibirrer', 'gilmanda'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.nfw_param import NFWParam
from astropy.cosmology import FlatLambdaCDM


class TestLensCosmo(object):
    """
    tests the UnitManager class routines
    """
    def setup(self):

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.nfwParam = NFWParam(cosmo=cosmo)
        self.z = 0.5  # needed fixed redshift for the inversion function

    def test_rho0_c(self):
        c = 4

        rho0 = self.nfwParam.rho0_c(c, z=self.z)
        c_out = self.nfwParam.c_rho0(rho0, z=self.z)
        npt.assert_almost_equal(c_out, c, decimal=3)

    def test_rhoc_z(self):
        z = 0
        rho0_z = self.nfwParam.rhoc_z(z=z)
        npt.assert_almost_equal(self.nfwParam.rhoc * (1+z)**3, rho0_z)

    def test_M200(self):
        M200 = self.nfwParam.M200(rs=1, rho0=1, c=1)
        npt.assert_almost_equal(M200, 2.4271590540348216, decimal=5)

    def test_profileMain(self):
        M = 10**(13.5)
        z = 0.5
        r200, rho0, c, Rs = self.nfwParam.nfw_Mz(M, z)

        c_ = self.nfwParam.c_M_z(M, z)
        r200_ = self.nfwParam.r200_M(M, z)
        rho0_ = self.nfwParam.rho0_c(c, z)
        Rs_ = r200_ / c_
        npt.assert_almost_equal(c_, c, decimal=5)
        npt.assert_almost_equal(r200_, r200, decimal=5)
        npt.assert_almost_equal(rho0_, rho0, decimal=5)
        npt.assert_almost_equal(Rs_, Rs, decimal=5)

    def test_against_colossus(self):
        """
        This test class asks to get the same parameters back as colossus: https://bdiemer.bitbucket.io/colossus/index.html
        """
        cosmo = FlatLambdaCDM(H0=70, Om0=0.285, Ob0=0.05)
        nfw_param = NFWParam(cosmo=cosmo)

        from colossus.cosmology import cosmology as cosmology_colossus
        from colossus.halo.profile_nfw import NFWProfile
        colossus_kwargs = {'H0': 70, 'Om0': 0.285, 'Ob0': 0.05, 'ns': 0.96, 'sigma8': 0.82}
        colossus = cosmology_colossus.setCosmology('custom', colossus_kwargs)

        m200 = 10 ** 8
        c = 17.

        zvals = np.linspace(0.0, 2, 50)
        h = 0.7

        for z in zvals:
            nfw_colossus = NFWProfile(m200 * h, z, mdef='200c')
            rhos_colossus, rs_colossus = nfw_colossus.fundamentalParameters(m200 * h, c, z, mdef='200c')
            r200_colossus = rs_colossus * c

            # according to colossus documentation the density is in physical units[M h^2/kpc^3] and distance [kpc/h]
            rs_colossus *= h ** -1
            rhos_colossus *= h ** 2


            r200_lenstronomy = nfw_param.r200_M(m200 * h, z) / h  # physical radius r200
            rs_lenstronomy = r200_lenstronomy / c
            rhos_lenstronomy = nfw_param.rho0_c(c, z) * h ** 2  # physical density in M_sun/Mpc**3

            # convert Mpc to kpc
            rhos_lenstronomy *= 1000 ** -3
            rs_lenstronomy *= 1000
            npt.assert_almost_equal(rs_lenstronomy/rs_colossus, 1, decimal=3)
            npt.assert_almost_equal(rhos_lenstronomy / rhos_colossus, 1, decimal=3)


if __name__ == '__main__':
    pytest.main()

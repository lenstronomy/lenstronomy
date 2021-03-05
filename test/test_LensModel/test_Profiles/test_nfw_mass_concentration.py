__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_mass_concentration import NFWMC
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

import numpy.testing as npt
import pytest


class TestNFWMC(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.z_lens, self.z_source = 0.5, 2
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.nfw = NFW()
        self.nfwmc = NFWMC(z_source=self.z_source, z_lens=self.z_lens, cosmo=cosmo)
        self.lensCosmo = LensCosmo(z_lens=self.z_lens, z_source=self.z_source, cosmo=cosmo)

    def test_function(self):
        x, y = 1., 1.
        logM = 12
        concentration = 10
        f_mc = self.nfwmc.function(x, y, logM, concentration, center_x=0, center_y=0)
        Rs, alpha_Rs = self.lensCosmo.nfw_physical2angle(10**logM, concentration)
        f_ = self.nfw.function(x, y, Rs, alpha_Rs, center_x=0, center_y=0)
        npt.assert_almost_equal(f_mc, f_, decimal=8)

    def test_derivatives(self):
        x, y = 1., 1.
        logM = 12
        concentration = 10
        f_x_mc, f_y_mc = self.nfwmc.derivatives(x, y, logM, concentration, center_x=0, center_y=0)
        Rs, alpha_Rs = self.lensCosmo.nfw_physical2angle(10 ** logM, concentration)
        f_x, f_y = self.nfw.derivatives(x, y, Rs, alpha_Rs, center_x=0, center_y=0)
        npt.assert_almost_equal(f_x_mc, f_x, decimal=8)
        npt.assert_almost_equal(f_y_mc, f_y, decimal=8)

    def test_hessian(self):
        x, y = 1., 1.
        logM = 12
        concentration = 10
        f_xx_mc, f_xy_mc, f_yx_mc, f_yy_mc = self.nfwmc.hessian(x, y, logM, concentration, center_x=0, center_y=0)
        Rs, alpha_Rs = self.lensCosmo.nfw_physical2angle(10 ** logM, concentration)
        f_xx, f_xy, f_yx, f_yy = self.nfw.hessian(x, y, Rs, alpha_Rs, center_x=0, center_y=0)
        npt.assert_almost_equal(f_xx_mc, f_xx, decimal=8)
        npt.assert_almost_equal(f_yy_mc, f_yy, decimal=8)
        npt.assert_almost_equal(f_xy_mc, f_xy, decimal=8)
        npt.assert_almost_equal(f_yx_mc, f_yx, decimal=8)

    def test_static(self):
        x, y = 1., 1.
        logM = 12
        concentration = 10
        f_ = self.nfwmc.function(x, y, logM, concentration, center_x=0, center_y=0)
        self.nfwmc.set_static(logM, concentration)
        f_static = self.nfwmc.function(x, y, 0, 0, center_x=0, center_y=0)
        npt.assert_almost_equal(f_, f_static, decimal=8)
        self.nfwmc.set_dynamic()
        f_dyn = self.nfwmc.function(x, y, 11, 20, center_x=0, center_y=0)
        assert f_dyn != f_static


if __name__ == '__main__':
    pytest.main()

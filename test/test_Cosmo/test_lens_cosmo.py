__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lens_cosmo import LensCosmo, FlatLCDM


class TestLensCosmo(object):
    """
    tests the UnitManager class routines
    """
    def setup(self):
        z_L = 0.8
        z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.lensCosmo = LensCosmo(z_L, z_S, cosmo=cosmo)

    def test_ang_dist(self):
        npt.assert_almost_equal(self.lensCosmo.D_s, 1588.9213590743666, decimal=8)
        npt.assert_almost_equal(self.lensCosmo.D_d, 1548.7055203661785, decimal=8)
        npt.assert_almost_equal(self.lensCosmo.D_ds, 892.0038749095863, decimal=8)

    def test_epsilon_crit(self):
        npt.assert_almost_equal(self.lensCosmo.epsilon_crit/1.9121e+15, 1, decimal=3)

    def test_arcsec2phys(self):
        arcsec = np.array([1, 2]) # pixel coordinate from center
        physcoord = self.lensCosmo.arcsec2phys_lens(arcsec)
        assert physcoord[0] == 0.0075083362428338641
        assert physcoord[1] == 0.015016672485667728

        physcoord = self.lensCosmo.arcsec2phys_source(arcsec)
        assert physcoord[0] == 0.007703308130864105
        assert physcoord[1] == 0.01540661626172821

    def test_phys2arcsec_lens(self):
        phys = 1.
        arc_sec = self.lensCosmo.phys2arcsec_lens(phys)
        phys_new = self.lensCosmo.arcsec2phys_lens(arc_sec)
        assert phys_new == phys

    def test_mass_in_phi_E(self):
        phi_E = 1.5
        mass = self.lensCosmo.mass_in_theta_E(phi_E)
        assert mass == 761967261292.6725

    def test_kappa2proj_mass(self):
        kappa = 0.5
        mass = self.lensCosmo.kappa2proj_mass(kappa)
        npt.assert_almost_equal(mass, kappa * self.lensCosmo.epsilon_crit, decimal=3)

    def test_mass_in_coin(self):
        theta_E = 1.
        m_coin = self.lensCosmo.mass_in_coin(theta_E)
        npt.assert_almost_equal(m_coin, 165279526936.52194, decimal=0)

    def test_D_dt_model(self):
        D_dt = self.lensCosmo.D_dt
        assert D_dt == 4965.660384441859


class TestFlatLCDM(object):
    def setup(self):
        self.cosmo = FlatLCDM(z_lens=0.5, z_source=1.5)

    def test_D_d(self):
        D_d = self.cosmo.D_d(H_0=70, Om0=0.3)
        assert D_d == 1259.0835972889377

    def test_D_s(self):
        D_s = self.cosmo.D_s(H_0=70, Om0=0.3)
        assert D_s == 1745.5423064934419

    def test_D_ds(self):
        D_ds = self.cosmo.D_ds(H_0=70, Om0=0.3)
        assert D_ds == 990.0921481200791

    def test_D_dt(self):
        D_dt = self.cosmo.D_dt(H_0=70, Om0=0.3)
        assert D_dt == 3329.665360925441


if __name__ == '__main__':
    pytest.main()

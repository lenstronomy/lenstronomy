__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lens_cosmo import LensCosmo


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

    def test_mass_in_phi_E(self):
        phi_E = 1.5
        mass = self.lensCosmo.mass_in_theta_E(phi_E)
        assert mass == 761967261292.6725


if __name__ == '__main__':
    pytest.main()
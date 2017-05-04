__author__ = 'sibirrer'

import numpy as np
import pytest

from lenstronomy.Cosmo.unit_manager import UnitManager


class TestUnitManager(object):
    """
    tests the UnitManager class routines
    """
    def setup(self):
        z_L = 0.8
        z_S = 3.0
        #initializing UnitManager
        self.unitManager = UnitManager(z_L, z_S)

    def test_arcsec2phys(self):
        arcsec = np.array([1, 2]) # pixel coordinate from center
        physcoord = self.unitManager.arcsec2phys_lens(arcsec)
        assert physcoord[0] == 0.0077212615411298754
        assert physcoord[1] == 0.015442523082259751

        physcoord = self.unitManager.arcsec2phys_source(arcsec)
        assert physcoord[0] == 0.0078887179899263752
        assert physcoord[1] == 0.01577743597985275

    def test_mass_in_phi_E(self):
        phi_E = 1.5
        mass = self.unitManager.mass_in_phi_E(phi_E)
        assert mass == 786148363450.92627

if __name__ == '__main__':
    pytest.main()
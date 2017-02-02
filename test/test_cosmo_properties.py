import pytest

from lenstronomy.Cosmo.cosmo_properties import CosmoProp
import lenstronomy.Cosmo.constants as const

class TestCosmoProp(object):

    def setup(self):
        self.z_L = 0.8
        self.z_S = 3.0
        self.cosmoProp = CosmoProp(self.z_L,self.z_S)

    def test_scale_factor(self):
        assert self.cosmoProp.a_L == 1./(1+self.z_L)
        assert self.cosmoProp.a_S == 1./(1+self.z_S)

    def test_a_z(self):
        assert self.cosmoProp.a_z(0.) == 1.

    def test_ang_dist(self):
        assert (self.cosmoProp.dist_OL < 1593) and (self.cosmoProp.dist_OL > 1592.5)
        assert (self.cosmoProp.dist_OS < 1628) and (self.cosmoProp.dist_OS > 1627)
        assert (self.cosmoProp.dist_LS < 910.5) and (self.cosmoProp.dist_LS > 910.4)

    def test_epsilon_crit(self):
        assert (self.cosmoProp.epsilon_crit < 1.9e+15) and (self.cosmoProp.epsilon_crit > 1.8e+15)

    def test_rho_crit(self):
        assert self.cosmoProp.rho_crit == 127430827515.61508

if __name__ == '__main__':
    pytest.main()
import pytest

from lenstronomy.Cosmo.background import Background


class TestCosmoProp(object):

    def setup(self):
        self.z_L = 0.8
        self.z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.bkg = Background(cosmo=cosmo)

    def test_scale_factor(self):
        z = 0.7
        assert self.bkg.a_z(z) == 1./(1+z)

    def test_rho_crit(self):
        assert self.bkg.rho_crit == 135955133951.10692


if __name__ == '__main__':
    pytest.main()
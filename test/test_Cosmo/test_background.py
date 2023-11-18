import pytest
import numpy.testing as npt

from lenstronomy.Cosmo.background import Background


class TestCosmoProp(object):
    def setup_method(self):
        self.z_L = 0.8
        self.z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.bkg = Background(cosmo=cosmo)

    def test_scale_factor(self):
        z = 0.7
        assert self.bkg.a_z(z) == 1.0 / (1 + z)

    def test_rho_crit(self):
        assert self.bkg.rho_crit == 135955133951.10692

    def test_interpol(self):
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        bkg = Background(cosmo=cosmo)

        bkg_interp = Background(cosmo=cosmo, interp=True, num_interp=100, z_stop=10)
        d_xy = bkg.d_xy(z_observer=0.1, z_source=0.8)
        d_xy_interp = bkg_interp.d_xy(z_observer=0.1, z_source=0.8)
        npt.assert_almost_equal(d_xy_interp / d_xy, 1, decimal=5)


if __name__ == "__main__":
    pytest.main()

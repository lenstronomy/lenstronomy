__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.nfw_vir_trunc import NFWVirTrunc
from lenstronomy.Util import util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

import numpy as np
import numpy.testing as npt
import pytest


class TestNFW(object):
    """Tests the Gaussian methods."""

    def setup_method(self):
        z_lens = 0.55
        z_source = 2.5
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.nfw = NFWVirTrunc(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        self.lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
        NFWVirTrunc(z_lens=z_lens, z_source=z_source, cosmo=None)

    def test_G(self):
        c = 10.0
        num = 1000
        l = 2 * c
        r = np.linspace(0, l, 1000)
        out = self.nfw._G(c, c=c)
        assert out == 0
        out = self.nfw._G(x=1, c=c)
        npt.assert_almost_equal(out, 0.32892146681210577, decimal=6)
        out = self.nfw._G(x=2, c=c)
        npt.assert_almost_equal(out, 0.12735521436564, decimal=6)

        out = self.nfw._G(x=c + 1, c=c)
        npt.assert_almost_equal(out, 0, decimal=6)
        kappa = self.nfw._G(r, c=c) * r * np.pi * 2
        kappa_int = np.sum(kappa) / num * l / c
        f = self.nfw._f(c)  # / self.nfw._f(c=1)

        # import matplotlib.pyplot as plt
        # plt.plot(r, kappa)
        # plt.show()
        # npt.assert_almost_equal(kappa_int, 1, decimal=1)
        # assert 1 == 0

    def test_kappa(self):
        c = 1.0
        logM = 13.0
        M = 10**logM
        theta_vir = self.nfw._lens_cosmo.nfw_M_theta_r200(M)
        print(theta_vir, "test theta_vir")
        print(theta_vir / c, "theta_Rs")

        num = 1000
        theta = np.linspace(0, theta_vir, num)
        d_theta = theta_vir / num

        kappa = self.nfw.kappa(theta, logM=logM, c=c) * theta * np.pi * 2 * d_theta
        f = self.nfw._f(c)
        print(f, "f")
        kappa_int = np.sum(kappa)
        mass = kappa_int * self.lensCosmo.sigma_crit_angle
        npt.assert_almost_equal(mass / M, 1, decimal=2)

    def test_radial_profile(self):
        r = np.logspace(start=-2, stop=2, num=100)
        c = 10
        logM = 13.0
        # kappa = self.nfw.kappa(r, logM=logM, c=c)
        import matplotlib.pyplot as plt

        # plt.loglog(r, kappa)
        # plt.show()
        # assert 1 == 0


if __name__ == "__main__":
    pytest.main()

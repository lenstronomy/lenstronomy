__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util  import util


class TestLensCosmo(object):
    """
    tests the UnitManager class routines
    """
    def setup_method(self):
        z_L = 0.8
        z_S = 3.0
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.lensCosmo = LensCosmo(z_L, z_S, cosmo=cosmo)

    def test_ang_dist(self):
        npt.assert_almost_equal(self.lensCosmo.ds, 1588.9213590743666, decimal=8)
        npt.assert_almost_equal(self.lensCosmo.dd, 1548.7055203661785, decimal=8)
        npt.assert_almost_equal(self.lensCosmo.dds, 892.0038749095863, decimal=8)

    def test_epsilon_crit(self):
        npt.assert_almost_equal(self.lensCosmo.sigma_crit / 1.9121e+15, 1, decimal=3)

    def test_arcsec2phys(self):
        arcsec = np.array([1, 2]) # pixel coordinate from center
        physcoord = self.lensCosmo.arcsec2phys_lens(arcsec)
        npt.assert_almost_equal(physcoord[0], 0.0075083362428338641, decimal=8)
        npt.assert_almost_equal(physcoord[1], 0.015016672485667728, decimal=8)

        physcoord = self.lensCosmo.arcsec2phys_source(arcsec)
        npt.assert_almost_equal(physcoord[0], 0.007703308130864105, decimal=8)
        npt.assert_almost_equal(physcoord[1], 0.01540661626172821, decimal=8)

    def test_phys2arcsec_lens(self):
        phys = 1.
        arc_sec = self.lensCosmo.phys2arcsec_lens(phys)
        phys_new = self.lensCosmo.arcsec2phys_lens(arc_sec)
        npt.assert_almost_equal(phys_new, phys, decimal=8)

    def test_mass_in_phi_E(self):
        phi_E = 1.5
        mass = self.lensCosmo.mass_in_theta_E(phi_E)
        npt.assert_almost_equal(mass, 761967261292.6725, decimal=2)

    def test_kappa2proj_mass(self):
        kappa = 0.5
        mass = self.lensCosmo.kappa2proj_mass(kappa)
        npt.assert_almost_equal(mass, kappa * self.lensCosmo.sigma_crit, decimal=3)

    def test_mass_in_coin(self):
        theta_E = 1.
        m_coin = self.lensCosmo.mass_in_coin(theta_E)
        npt.assert_almost_equal(m_coin, 165279526936.52194, decimal=0)

    def test_D_dt_model(self):
        D_dt = self.lensCosmo.ddt
        npt.assert_almost_equal(D_dt, 4965.660384441859, decimal=8)

    def test_nfw_angle2physical(self):
        Rs_angle = 6.
        alpha_Rs = 1.
        rho0, Rs, c, r200, M200 = self.lensCosmo.nfw_angle2physical(Rs_angle, alpha_Rs)
        assert Rs * c == r200

    def test_nfw_physical2angle(self):
        M = 10.**13.5
        c = 4
        Rs_angle, alpha_Rs = self.lensCosmo.nfw_physical2angle(M, c)
        rho0, Rs, c_out, r200, M200 = self.lensCosmo.nfw_angle2physical(Rs_angle, alpha_Rs)
        npt.assert_almost_equal(c_out, c, decimal=3)
        npt.assert_almost_equal(np.log10(M200), np.log10(M), decimal=4)

    def test_sis_theta_E2sigma_v(self):
        theta_E = 2.
        sigma_v = self.lensCosmo.sis_theta_E2sigma_v(theta_E)
        theta_E_out = self.lensCosmo.sis_sigma_v2theta_E(sigma_v)
        npt.assert_almost_equal(theta_E_out, theta_E, decimal=5)

    def test_fermat2delays(self):

        fermat_pot = 0.5
        dt_days = self.lensCosmo.time_delay_units(fermat_pot)
        fermat_pot_out = self.lensCosmo.time_delay2fermat_pot(dt_days)
        npt.assert_almost_equal(fermat_pot, fermat_pot_out, decimal=10)

    def test_uldm_angular2phys(self):

        kappa_0, theta_c = 0.1, 3
        mlog10, Mlog10 = self.lensCosmo.uldm_angular2phys(kappa_0, theta_c)
        npt.assert_almost_equal(mlog10, -24.3610006, decimal=5)
        npt.assert_almost_equal(Mlog10, 11.7195843, decimal=5)

    def test_uldm_mphys2angular(self):

        m_log10, M_log10 = -24, 11
        kappa_0, theta_c = self.lensCosmo.uldm_mphys2angular(m_log10, M_log10)
        mcheck, Mcheck = self.lensCosmo.uldm_angular2phys(kappa_0, theta_c)
        npt.assert_almost_equal(mcheck, m_log10, decimal=4)
        npt.assert_almost_equal(Mcheck, M_log10, decimal=4)

    def test_a_z(self):

        a = self.lensCosmo.background.a_z(z=1)
        npt.assert_almost_equal(a, 0.5)

    def test_sersic_m_star2k_eff(self):
        m_star = 10**11.5
        R_sersic = 1
        n_sersic = 4
        k_eff = self.lensCosmo.sersic_m_star2k_eff(m_star, R_sersic, n_sersic)
        npt.assert_almost_equal(k_eff, 0.1294327891669961, decimal=5)

        m_star_out = self.lensCosmo.sersic_k_eff2m_star(k_eff, R_sersic, n_sersic)
        npt.assert_almost_equal(m_star_out, m_star, decimal=6)

    def test_hernquist_angular2phys(self):
        m_star = 10**10  # in M_sun
        rs = 0.01  # in Mpc

        # test bijective transformation
        sigma0, rs_angle = self.lensCosmo.hernquist_phys2angular(mass=m_star, rs=rs)
        m_star_new, rs_new = self.lensCosmo.hernquist_angular2phys(sigma0=sigma0, rs_angle=rs_angle)
        npt.assert_almost_equal(m_star_new, m_star, decimal=1)
        npt.assert_almost_equal(rs_new, rs, decimal=8)

    def test_hernquist_mass_normalization(self):
        m_star = 10 ** 10  # in M_sun
        rs = 0.01  # in Mpc

        # test bijective transformation
        sigma0, rs_angle = self.lensCosmo.hernquist_phys2angular(mass=m_star, rs=rs)

        # test mass integrals
        # make large grid
        delta_pix = rs_angle / 30.
        x, y = util.make_grid(numPix=501, deltapix=delta_pix)
        # compute convergence
        from lenstronomy.LensModel.lens_model import LensModel
        lens_model = LensModel(lens_model_list=['HERNQUIST'])
        kwargs = [{'sigma0': sigma0, 'Rs': rs_angle, 'center_x': 0, 'center_y': 0}]
        kappa = lens_model.kappa(x, y, kwargs)
        # sum up convergence
        kappa_tot = np.sum(kappa) * delta_pix ** 2
        # transform to mass
        mass_tot = kappa_tot * self.lensCosmo.sigma_crit_angle

        # compare
        npt.assert_almost_equal(mass_tot/ m_star, 1, decimal=1)


if __name__ == '__main__':
    pytest.main()

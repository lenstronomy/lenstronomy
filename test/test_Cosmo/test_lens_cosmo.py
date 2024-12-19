__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util import util


class TestLensCosmo(object):
    """Tests the UnitManager class routines."""

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
        npt.assert_almost_equal(self.lensCosmo.sigma_crit / 1.9121e15, 1, decimal=3)

    def test_arcsec2phys(self):
        arcsec = np.array([1, 2])  # pixel coordinate from center
        physcoord = self.lensCosmo.arcsec2phys_lens(arcsec)
        npt.assert_almost_equal(physcoord[0], 0.0075083362428338641, decimal=8)
        npt.assert_almost_equal(physcoord[1], 0.015016672485667728, decimal=8)

        physcoord = self.lensCosmo.arcsec2phys_source(arcsec)
        npt.assert_almost_equal(physcoord[0], 0.007703308130864105, decimal=8)
        npt.assert_almost_equal(physcoord[1], 0.01540661626172821, decimal=8)

    def test_phys2arcsec_lens(self):
        phys = 1.0
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
        theta_E = 1.0
        m_coin = self.lensCosmo.mass_in_coin(theta_E)
        npt.assert_almost_equal(m_coin, 165279526936.52194, decimal=0)

    def test_D_dt_model(self):
        D_dt = self.lensCosmo.ddt
        npt.assert_almost_equal(D_dt, 4965.660384441859, decimal=8)

    def test_nfw_angle2physical(self):
        Rs_angle = 6.0
        alpha_Rs = 1.0
        rho0, Rs, c, r200, M200 = self.lensCosmo.nfw_angle2physical(Rs_angle, alpha_Rs)
        assert Rs * c == r200

    def test_nfw_physical2angle(self):
        M = 10.0**13.5
        c = 4
        Rs_angle, alpha_Rs = self.lensCosmo.nfw_physical2angle(M, c)
        rho0, Rs, c_out, r200, M200 = self.lensCosmo.nfw_angle2physical(
            Rs_angle, alpha_Rs
        )
        npt.assert_almost_equal(c_out, c, decimal=3)
        npt.assert_almost_equal(np.log10(M200), np.log10(M), decimal=4)

    def test_gnfw_angle2physical(self):
        Rs_angle = 6.0
        alpha_Rs = 1.0
        gamma_in = 1

        rho0, Rs, c, r200, M200 = self.lensCosmo.gnfw_angle2physical(
            Rs_angle, alpha_Rs, gamma_in
        )
        assert Rs * c == r200
        rho0_nfw, Rs_nfw, c_nfw, r200_nfw, M200_nfw = self.lensCosmo.nfw_angle2physical(
            Rs_angle, alpha_Rs
        )
        npt.assert_almost_equal(rho0 / rho0_nfw, 1, decimal=8)
        npt.assert_almost_equal(Rs / Rs_nfw, 1, decimal=8)
        npt.assert_almost_equal(c / c_nfw, 1, decimal=8)
        npt.assert_almost_equal(r200 / r200_nfw, 1, decimal=8)
        npt.assert_almost_equal(M200 / M200_nfw, 1, decimal=8)

    def test_gnfw_physical2angle(self):
        M = 10.0**13.5
        c = 4
        Rs_angle, alpha_Rs = self.lensCosmo.gnfw_physical2angle(M, c, gamma_in=1)
        rho0, Rs, c_out, r200, M200 = self.lensCosmo.gnfw_angle2physical(
            Rs_angle, alpha_Rs, gamma_in=1
        )
        npt.assert_almost_equal(c_out, c, decimal=3)
        npt.assert_almost_equal(np.log10(M200), np.log10(M), decimal=4)

        Rs_angle_nfw, alpha_Rs_nfw = self.lensCosmo.nfw_physical2angle(M, c)

        npt.assert_almost_equal(Rs_angle / Rs_angle_nfw, 1, decimal=8)
        npt.assert_almost_equal(alpha_Rs / alpha_Rs_nfw, 1, decimal=8)

    def test_gnfwParam_physical(self):
        M = 10.0**13.5
        c = 10
        rh0, Rs, r200 = self.lensCosmo.gnfwParam_physical(M, c, gamma_in=1)

        rho0_nfw, Rs_nfw, r200_nfw = self.lensCosmo.nfwParam_physical(M, c)

        npt.assert_almost_equal(rho0_nfw / rh0, 1, decimal=10)
        npt.assert_almost_equal(Rs_nfw / Rs, 1, decimal=10)

    def test_gnfw_M_theta_r200(self):
        M = 10.0**13.5

        theta200 = self.lensCosmo.gnfw_M_theta_r200(M)
        theta200_nfw = self.lensCosmo.nfw_M_theta_r200(M)

        npt.assert_almost_equal(theta200 / theta200_nfw, 1, decimal=10)

    def test_sis_theta_E2sigma_v(self):
        theta_E = 2.0
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
        m_star_new, rs_new = self.lensCosmo.hernquist_angular2phys(
            sigma0=sigma0, rs_angle=rs_angle
        )
        npt.assert_almost_equal(m_star_new, m_star, decimal=1)
        npt.assert_almost_equal(rs_new, rs, decimal=8)

    def test_hernquist_mass_normalization(self):
        m_star = 10**10  # in M_sun
        rs = 0.01  # in Mpc

        # test bijective transformation
        sigma0, rs_angle = self.lensCosmo.hernquist_phys2angular(mass=m_star, rs=rs)

        # test mass integrals
        # make large grid
        delta_pix = rs_angle / 30.0
        x, y = util.make_grid(numPix=501, deltapix=delta_pix)
        # compute convergence
        from lenstronomy.LensModel.lens_model import LensModel

        lens_model = LensModel(lens_model_list=["HERNQUIST"])
        kwargs = [{"sigma0": sigma0, "Rs": rs_angle, "center_x": 0, "center_y": 0}]
        kappa = lens_model.kappa(x, y, kwargs)
        # sum up convergence
        kappa_tot = np.sum(kappa) * delta_pix**2
        # transform to mass
        mass_tot = kappa_tot * self.lensCosmo.sigma_crit_angle

        # compare
        npt.assert_almost_equal(mass_tot / m_star, 1, decimal=1)

    def test_vel_disp_dPIED_sigma0(self):
        from lenstronomy.LensModel.lens_model import LensModel
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensCosmo = LensCosmo(z_lens=2, z_source=5, cosmo=cosmo)

        vel_disp = 250
        # make the test such that dPIED effectively mimics a SIS profile
        theta_E_sis = lensCosmo.sis_sigma_v2theta_E(v_sigma=vel_disp)
        sis = LensModel(lens_model_list=["SIS"])
        kwargs_sis = [{"theta_E": theta_E_sis}]

        lens_model = LensModel(lens_model_list=["PJAFFE"])
        r = np.logspace(start=-2, stop=1, num=100)

        Rs = 100000
        Ra = 0.00001
        Ra_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        for Ra in Ra_list:
            sigma0 = lensCosmo.vel_disp_dPIED_sigma0(vel_disp, Ra=Ra, Rs=Rs)
            kwargs_lens = [
                {"sigma0": sigma0, "Ra": Ra, "Rs": Rs, "center_x": 0, "center_y": 0}
            ]

            # plt.semilogx(r, lens_model.kappa(r, 0, kwargs_lens) / sis.kappa(r, 0, kwargs_sis), label=Ra)
        # plt.legend()
        # plt.show()

        # calculate Einstein radius and compare it with SIS profile
        from lenstronomy.Analysis.lens_profile import LensProfileAnalysis

        lens_analysis = LensProfileAnalysis(lens_model=lens_model)
        theta_E_dPIED = lens_analysis.effective_einstein_radius(kwargs_lens=kwargs_lens)
        npt.assert_almost_equal(theta_E_dPIED / theta_E_sis, 1, decimal=2)

    def test_beta_double_source_plane(self):
        beta = self.lensCosmo.beta_double_source_plane(
            z_lens=0.5, z_source_1=1, z_source_2=2
        )
        beta_true = self.lensCosmo.background.beta_double_source_plane(
            z_lens=0.5, z_source_1=1, z_source_2=2
        )
        npt.assert_almost_equal(beta, beta_true, decimal=5)

    def test_theta_E_power_law_scaling(self):
        theta_E_convention = 1
        kappa_ext_convention = 0.0
        gamma_pl = 2.3
        z_lens = 0.5
        z_source_convention = 5
        z_source = 1
        theta_E_conversion = self.lensCosmo.theta_E_power_law_scaling(
            theta_E_convention,
            kappa_ext_convention,
            gamma_pl,
            z_lens,
            z_source_convention,
            z_source,
        )
        # numerical solution for the Einstein radius
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.Analysis.lens_profile import LensProfileAnalysis

        lens_model = LensModel(
            lens_model_list=["EPL", "CONVERGENCE"],
            z_lens=z_lens,
            z_source_convention=z_source_convention,
            multi_plane=False,
            z_source=z_source,
        )
        kwargs_lens = [
            {
                "theta_E": theta_E_convention,
                "gamma": gamma_pl,
                "e1": 0,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"kappa": kappa_ext_convention},
        ]

        lens_analysis = LensProfileAnalysis(lens_model=lens_model)
        theta_E = lens_analysis.effective_einstein_radius(
            kwargs_lens, r_min=1e-5, r_max=5e1, num_points=100
        )
        npt.assert_almost_equal(theta_E_conversion, theta_E, decimal=3)
        # and here we test no alterations
        theta_E_conversion = self.lensCosmo.theta_E_power_law_scaling(
            theta_E_convention,
            0,
            gamma_pl,
            z_lens,
            z_source_convention,
            z_source_convention,
        )
        npt.assert_almost_equal(theta_E_conversion, theta_E_convention, decimal=8)

        theta_E_convention = 1
        kappa_ext_convention = 0.2
        gamma_pl = 2
        z_lens = 0.5
        z_source_convention = 5
        z_source = 1
        theta_E_conversion = self.lensCosmo.theta_E_power_law_scaling(
            theta_E_convention,
            kappa_ext_convention,
            gamma_pl,
            z_lens,
            z_source_convention,
            z_source,
        )
        # numerical solution for the Einstein radius
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.Analysis.lens_profile import LensProfileAnalysis

        lens_model = LensModel(
            lens_model_list=["EPL", "CONVERGENCE"],
            z_lens=z_lens,
            z_source_convention=z_source_convention,
            multi_plane=False,
            z_source=z_source,
        )
        kwargs_lens = [
            {
                "theta_E": theta_E_convention,
                "gamma": gamma_pl,
                "e1": 0,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"kappa": kappa_ext_convention},
        ]

        lens_analysis = LensProfileAnalysis(lens_model=lens_model)
        theta_E = lens_analysis.effective_einstein_radius(
            kwargs_lens, r_min=1e-5, r_max=5e1, num_points=100
        )
        npt.assert_almost_equal(theta_E_conversion, theta_E, decimal=3)
        # and here we test no alterations
        theta_E_conversion = self.lensCosmo.theta_E_power_law_scaling(
            theta_E_convention,
            0,
            gamma_pl,
            z_lens,
            z_source_convention,
            z_source_convention,
        )
        npt.assert_almost_equal(theta_E_conversion, theta_E_convention, decimal=8)


if __name__ == "__main__":
    pytest.main()

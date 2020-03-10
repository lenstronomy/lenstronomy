__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
import numpy as np

from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util import constants as const


class TestTDCosmography(object):

    def setup(self):
        kwargs_model = {'lens_light_model_list': ['HERNQUIST'],
                        'lens_model_list': ['SIE'],
                        'point_source_model_list': ['LENSED_POSITION']}
        z_lens = 0.5
        z_source = 2.5
        TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=None, lens_model_kinematics_bool=None,
                      light_model_kinematics_bool=None)
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        self.td_cosmo = TDCosmography(z_lens, z_source, kwargs_model, cosmo_fiducial=cosmo, lens_model_kinematics_bool=None,
                 light_model_kinematics_bool=None)
        self.lens = LensModel(lens_model_list=['SIE'], cosmo=cosmo, z_lens=z_lens, z_source=z_source)
        self.solver = LensEquationSolver(lensModel=self.lens)

        self.kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': -0.2, 'center_x': 0, 'center_y': 0}]
        source_x, source_y = 0, 0.05
        image_x, image_y = self.solver.image_position_from_source(source_x, source_y, self.kwargs_lens, min_distance=0.1, search_window=10)
        self.kwargs_ps = [{'ra_image': image_x, 'dec_image': image_y}]
        self.image_x, self.image_y = image_x, image_y

    def test_time_delays(self):
        dt = self.td_cosmo.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=0)
        dt_true = self.lens.arrival_time(self.image_x, self.image_y, self.kwargs_lens)
        npt.assert_almost_equal(dt, dt_true, decimal=6)

    def test_fermat_potential(self):
        fermat_pot = self.td_cosmo.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        fermat_pot_true = self.lens.fermat_potential(self.image_x, self.image_y, self.kwargs_lens)
        npt.assert_almost_equal(fermat_pot, fermat_pot_true, decimal=6)

        diff = 0.1
        kwargs_ps = [{'ra_image': self.image_x+diff, 'dec_image': self.image_y}]
        fermat_pot = self.td_cosmo.fermat_potential(self.kwargs_lens, kwargs_ps)
        fermat_pot_true = self.lens.fermat_potential(self.image_x+diff, self.image_y, self.kwargs_lens)
        ratio = fermat_pot / fermat_pot_true
        assert np.max(np.abs(ratio)) > 1.05

    def test_cosmo_inference(self):

        # set up a cosmology
        # compute image postions
        # compute J and velocity dispersion
        D_dt = self.td_cosmo._lens_cosmo.ddt
        D_d = self.td_cosmo._lens_cosmo.dd
        D_s = self.td_cosmo._lens_cosmo.ds
        D_ds = self.td_cosmo._lens_cosmo.dds
        fermat_potential_list = self.td_cosmo.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        dt_list = self.td_cosmo.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=0)
        dt = dt_list[0] - dt_list[1]
        d_fermat = fermat_potential_list[0] - fermat_potential_list[1]

        D_dt_infered = self.td_cosmo.ddt_from_time_delay(d_fermat_model=d_fermat, dt_measured=dt)
        npt.assert_almost_equal(D_dt_infered, D_dt, decimal=5)
        r_eff = 0.5
        kwargs_lens_light = [{'Rs': r_eff * 0.551, 'center_x': 0, 'center_y': 0}]
        kwargs_anisotropy = {'r_ani': 1}

        R_slit = 3.8
        dR_slit = 1.
        aperture_type = 'slit'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'width': dR_slit, 'length': R_slit,
                           'angle': 0, 'center_dec': 0}
        psf_fwhm = 0.7
        kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        self.td_cosmo.kinematic_observation_settings(kwargs_aperture, kwargs_seeing)

        anisotropy_model = 'OM'
        kwargs_numerics_galkin = {'interpol_grid_num': 500, 'log_integration': True,
                                  'max_integrate': 10, 'min_integrate': 0.001}
        self.td_cosmo.kinematics_modeling_settings(anisotropy_model, kwargs_numerics_galkin, analytic_kinematics=True,
                                             Hernquist_approx=False, MGE_light=False, MGE_mass=False)

        J = self.td_cosmo.velocity_dispersion_dimension_less(self.kwargs_lens, kwargs_lens_light, kwargs_anisotropy, r_eff=r_eff,
                                           theta_E=self.kwargs_lens[0]['theta_E'], gamma=2)

        J_map = self.td_cosmo.velocity_dispersion_map_dimension_less(self.kwargs_lens, kwargs_lens_light,
                                                                     kwargs_anisotropy, r_eff=r_eff,
                                                                     theta_E=self.kwargs_lens[0]['theta_E'], gamma=2)
        assert len(J_map) == 1
        npt.assert_almost_equal(J_map[0]/J, 1, decimal=1)
        sigma_v2 = J * D_s/D_ds * const.c ** 2
        sigma_v = np.sqrt(sigma_v2) / 1000.  # convert to [km/s]
        print(sigma_v, 'test sigma_v')
        Ds_Dds = self.td_cosmo.ds_dds_from_kinematics(sigma_v, J, kappa_s=0, kappa_ds=0)
        npt.assert_almost_equal(Ds_Dds, D_s/D_ds)

        # now we perform a mass-sheet transform in the observables but leave the models identical with a convergence correction
        kappa_s = 0.5
        dt_list = self.td_cosmo.time_delays(self.kwargs_lens, self.kwargs_ps, kappa_ext=kappa_s)
        sigma_v_kappa = sigma_v * np.sqrt(1-kappa_s)
        dt = dt_list[0] - dt_list[1]
        D_dt_infered, D_d_infered = self.td_cosmo.ddt_dd_from_time_delay_and_kinematics(d_fermat_model=d_fermat, dt_measured=dt,
                                                                                        sigma_v_measured=sigma_v_kappa, J=J, kappa_s=kappa_s,
                                                                                        kappa_ds=0, kappa_d=0)
        npt.assert_almost_equal(D_dt_infered, D_dt, decimal=6)
        npt.assert_almost_equal(D_d_infered, D_d, decimal=6)


if __name__ == '__main__':
    pytest.main()

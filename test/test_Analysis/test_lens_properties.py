__author__ = 'sibirrer'

import numpy.testing as npt
import pytest

from lenstronomy.Analysis.lens_properties import LensProp
import lenstronomy.Util.param_util as param_util


class TestLensProp(object):

    def setup(self):
        pass

    def test_velocity_dispersion_new(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP', 'SHEAR', 'SIS', 'SIS', 'SIS'],
                         'foreground_shear': False, 'lens_model_deflector_bool': [True, False, False, False, False],
                          'lens_light_deflector_bool': [True], 'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC']}
        lensProp = LensProp(z_lens, z_source, kwargs_options)
        kwargs_lens = [{'theta_E': 1.4272358196260446, 'e1': 0, 'center_x': -0.044798916793300093, 'center_y': 0.0054408937891703788, 'e2': 0, 'gamma': 1.8},
                       {'e1': -0.050871696555354479, 'e2': -0.0061601733920590464}, {'center_y': 2.79985456, 'center_x': -2.32019894,
                        'theta_E': 0.28165274714097904}, {'center_y': 3.83985426,
                        'center_x': -2.32019933, 'theta_E': 0.0038110812674654873},
                       {'center_y': 4.31985428, 'center_x': -1.68019931, 'theta_E': 0.45552039839735037}]

        phi, q = -0.52624727893702705, 0.79703498156919605
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_light = [{'n_sersic': 1.1212528655709217,
                              'center_x': -0.019674496231393473,
                              'e1': e1, 'e2': e2, 'amp': 1.1091367792010356, 'center_y': 0.076914975081560991,
                               'R_sersic': 0.42691611878867058},
                             {'R_sersic': 0.03025682660635394, 'amp': 139.96763298885992, 'n_sersic': 1.90000008624093865,
                              'center_x': -0.019674496231393473, 'center_y': 0.076914975081560991}]
        r_ani = 0.62
        kwargs_anisotropy = {'r_ani': r_ani}
        R_slit = 3.8
        dR_slit = 1.
        kwargs_aperture = {'center_ra': 0, 'width': dR_slit, 'length': R_slit, 'angle': 0, 'center_dec': 0}
        aperture_type = 'slit'
        psf_fwhm = 0.7
        anisotropy_model = 'OsipkovMerritt'
        r_eff = 0.211919902322

        v_sigma = lensProp.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                         kwargs_aperture, psf_fwhm, aperture_type, anisotropy_model,
                                                         MGE_light=True, r_eff=r_eff, lens_model_kinematics_bool=kwargs_options['lens_model_deflector_bool'])
        v_sigma_mge_lens = lensProp.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
                                                                  psf_fwhm, aperture_type, anisotropy_model, MGE_light=True, MGE_mass=True,
                                                                  r_eff=r_eff, lens_model_kinematics_bool=kwargs_options['lens_model_deflector_bool'])
        vel_disp_temp = lensProp.velocity_dispersion(kwargs_lens, kwargs_lens_light, aniso_param=r_ani, r_eff=r_eff, R_slit=R_slit, dR_slit=dR_slit, psf_fwhm=psf_fwhm, num_evaluate=5000)
        print(v_sigma, vel_disp_temp)
        #assert 1 == 0
        npt.assert_almost_equal(v_sigma / vel_disp_temp, 1, decimal=1)
        npt.assert_almost_equal(v_sigma_mge_lens / v_sigma, 1, decimal=1)

    def test_time_delays(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP'], 'point_source_model_list': ['LENSED_POSITION']}
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2}]
        kwargs_else = [{'ra_image': [-1, 0, 1], 'dec_image': [0, 0, 0]}]

        lensProp = LensProp(z_lens, z_source, kwargs_options)
        delays = lensProp.time_delays(kwargs_lens, kwargs_ps=kwargs_else, kappa_ext=0)
        npt.assert_almost_equal(delays[0], -31.710641699405745, decimal=8)
        npt.assert_almost_equal(delays[1], 0, decimal=8)
        npt.assert_almost_equal(delays[2], -31.710641699405745, decimal=8)

    def test_angular_diameter_relations(self):
        z_lens = 0.5
        z_source = 1.5
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = LensProp(z_lens, z_source, kwargs_model={}, cosmo=cosmo)
        sigma_v_model = 290
        sigma_v = 310
        kappa_ext = 0
        D_dt_model = 3000

        D_d, Ds_Dds = lensProp.angular_diameter_relations(sigma_v_model, sigma_v, kappa_ext, D_dt_model)
        npt.assert_almost_equal(D_d, 992.768, decimal=1)
        npt.assert_almost_equal(Ds_Dds, 2.01, decimal=2)

    def test_angular_distances(self):
        z_lens = 0.5
        z_source = 1.5
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
        lensProp = LensProp(z_lens, z_source, kwargs_model={}, cosmo=cosmo)
        sigma_v_measured = 290
        sigma_v_modeled = 310
        kappa_ext = 0
        time_delay_measured = 111
        fermat_pot = 0.7
        Ds_Dds, DdDs_Dds = lensProp.angular_distances(sigma_v_measured, time_delay_measured, kappa_ext, sigma_v_modeled,
                                                      fermat_pot)
        print(Ds_Dds, DdDs_Dds)
        npt.assert_almost_equal(Ds_Dds, 1.5428, decimal=2)
        npt.assert_almost_equal(DdDs_Dds, 3775.442, decimal=1)


if __name__ == '__main__':
    pytest.main()

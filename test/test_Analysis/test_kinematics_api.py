__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
import unittest

from lenstronomy.Analysis.kinematics_api import KinematicAPI
import lenstronomy.Util.param_util as param_util


class TestKinematicsAPI(object):

    def setup(self):
        pass

    def test_velocity_dispersion(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP', 'SHEAR', 'SIS', 'SIS', 'SIS'],

                          'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC']}
        kinematicAPI = KinematicAPI(z_lens, z_source, kwargs_options)
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
        aperture_type = 'slit'
        kwargs_aperture = {'aperture_type': aperture_type, 'center_ra': 0, 'width': dR_slit, 'length': R_slit, 'angle': 0, 'center_dec': 0}

        psf_fwhm = 0.7
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        anisotropy_model = 'OsipkovMerritt'
        kwargs_mge = {'n_comp': 20}
        r_eff = 0.211919902322

        v_sigma = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                         kwargs_aperture, kwargs_psf, anisotropy_model,
                                                         MGE_light=True, r_eff=r_eff,  kwargs_mge_light=kwargs_mge,
                                                         lens_model_kinematics_bool=[True, False, False, False, False])
        v_sigma_mge_lens = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy, kwargs_aperture,
                                                                  kwargs_psf, anisotropy_model, MGE_light=True, MGE_mass=True,
                                                                  kwargs_mge_light=kwargs_mge, kwargs_mge_mass=kwargs_mge,
                                                                  r_eff=r_eff, lens_model_kinematics_bool=[True, False, False, False, False])
        v_sigma_hernquist = kinematicAPI.velocity_dispersion_numerical(kwargs_lens, kwargs_lens_light, kwargs_anisotropy,
                                                                  kwargs_aperture, kwargs_psf, anisotropy_model,
                                                                  MGE_light=False, MGE_mass=False,
                                                                  r_eff=r_eff, Hernquist_approx=True,
                                                                  lens_model_kinematics_bool=[True, False, False, False, False])
        vel_disp_temp = kinematicAPI.velocity_dispersion(kwargs_lens, aniso_param=r_ani/r_eff, r_eff=r_eff,
                                                     kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                                                     num_evaluate=5000)
        print(v_sigma, vel_disp_temp)
        #assert 1 == 0
        npt.assert_almost_equal(v_sigma / vel_disp_temp, 1, decimal=1)
        npt.assert_almost_equal(v_sigma_mge_lens / v_sigma, 1, decimal=1)
        npt.assert_almost_equal(v_sigma / v_sigma_hernquist, 1, decimal=1)

    def test_kinematic_light_profile(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'SERSIC']}
        kwargs_mge = {'n_comp': 20}
        kinematicAPI = KinematicAPI(z_lens, z_source, kwargs_options)
        r_eff = 0.2
        kwargs_lens_light = [{'amp': 1, 'Rs': r_eff * 0.551, 'e1': 0., 'e2': 0, 'center_x': 0, 'center_y': 0},
                             {'amp': 1, 'R_sersic': 1, 'n_sersic': 2, 'center_x': -10, 'center_y': -10}]
        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=True,
                                                                                r_eff=r_eff,
                                                                                model_kinematics_bool=[True, False],
                                                                                kwargs_mge=kwargs_mge)
        assert light_profile_list[0] == 'MULTI_GAUSSIAN'

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=False,
                                                                                r_eff=r_eff, model_kinematics_bool=[True, False])
        assert light_profile_list[0] == 'HERNQUIST_ELLIPSE'

        light_profile_list, kwargs_light = kinematicAPI.kinematic_light_profile(kwargs_lens_light, MGE_fit=False,
                                                                                Hernquist_approx=True, r_eff=r_eff,
                                                                                model_kinematics_bool=[True, False])
        assert light_profile_list[0] == 'HERNQUIST'
        npt.assert_almost_equal(kwargs_light[0]['Rs'] / kwargs_lens_light[0]['Rs'], 1, decimal=2)

    def test_kinematic_lens_profiles(self):
        z_lens = 0.5
        z_source = 1.5
        kwargs_options = {'lens_model_list': ['SPEP', 'SHEAR']}
        lensProp = KinematicAPI(z_lens, z_source, kwargs_options)
        kwargs_lens = [{'theta_E': 1.4272358196260446, 'e1': 0, 'center_x': -0.044798916793300093,
                        'center_y': 0.0054408937891703788, 'e2': 0, 'gamma': 1.8},
                       {'e1': -0.050871696555354479, 'e2': -0.0061601733920590464}
                       ]

        kwargs_mge = {'n_comp': 20}
        mass_profile_list, kwargs_profile = lensProp.kinematic_lens_profiles(kwargs_lens, MGE_fit=True,
                                                                             kwargs_mge=kwargs_mge,
                                                                             model_kinematics_bool=[True, False])
        assert mass_profile_list[0] == 'MULTI_GAUSSIAN_KAPPA'

        mass_profile_list, kwargs_profile = lensProp.kinematic_lens_profiles(kwargs_lens, MGE_fit=False,
                                                                             model_kinematics_bool=[True, False])
        assert mass_profile_list[0] == 'SPEP'


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            kwargs_options = {'lens_light_model_list': ['HERNQUIST']}
            kinematicAPI = KinematicAPI(z_lens, z_source, kwargs_options)
            kwargs_light = [{'Rs': 1, 'amp': 1, 'center_x': 0, 'center_y': 0}]
            kinematicAPI.kinematic_light_profile(kwargs_light, MGE_fit=False,
                                                 Hernquist_approx=True, r_eff=None, model_kinematics_bool=[True])
            raise ValueError()


if __name__ == '__main__':
    pytest.main()

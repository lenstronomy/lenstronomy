"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import numpy as np
import scipy.integrate as integrate
from lenstronomy.GalKin.galkin_old import GalKinAnalytic
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics
import lenstronomy.Util.param_util as param_util


class TestGalkin(object):

    def setup(self):
        pass

    def test_galkin_vs_LOS_dispersion(self):
        """
        tests whether the old and new version provide the same answer
        :return:
        """
        # light profile
        light_profile = 'Hernquist'
        r_eff = 0.5
        kwargs_light = {'r_eff': r_eff}  # effective half light radius (2d projected) in arcsec

        # mass profile
        mass_profile = 'power_law'
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = {'theta_E': theta_E, 'gamma': gamma}  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'r_ani'
        r_ani = 0.5
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as shell
        #aperture_type = 'shell'
        #kwargs_aperture_inner = {'r_in': 0., 'r_out': 0.2, 'center_dec': 0, 'center_ra': 0}

        #kwargs_aperture_outer = {'r_in': 0., 'r_out': 1.5, 'center_dec': 0, 'center_ra': 0}

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 0.1  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}

        galkin = GalKinAnalytic(aperture=aperture_type, mass_profile=mass_profile, light_profile=light_profile,
                                anisotropy_type=anisotropy_type, psf_fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_aperture, kwargs_light, kwargs_anisotropy, num=2000)

        los_disp = AnalyticKinematics(**kwargs_cosmo)
        sigma_v2 = los_disp.vel_disp(gamma, theta_E, r_eff, r_ani=r_ani, R_slit=length, dR_slit=width,
                                     FWHM=psf_fwhm, rendering_number=2000)
        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=2)

    def test_log_linear_integral(self):
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = .5
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        kwargs_numerics_linear = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': False,
                           'max_integrate': 10, 'min_integrate': 0.001}
        kwargs_numerics_log = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': True,
                           'max_integrate': 10, 'min_integrate': 0.001}
        galkin_linear = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type,
                        anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_linear)
        galkin_log = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type,
                        anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_log)
        R = np.linspace(0.05, 1, 100)
        lin_I_R = np.zeros_like(R)
        log_I_R = np.zeros_like(R)
        for i in range(len(R)):
            lin_I_R[i] = galkin_linear.I_R_simga2(R[i], kwargs_profile, kwargs_light, kwargs_anisotropy)
            log_I_R[i] = galkin_log.I_R_simga2(R[i], kwargs_profile, kwargs_light, kwargs_anisotropy)
        print(log_I_R/lin_I_R)
        for i in range(len(R)):
            npt.assert_almost_equal(log_I_R[i] / lin_I_R[i], 1, decimal=2)

    def test_log_vs_linear_integral(self):
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = .5
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        kwargs_numerics_log = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': True,
                           'max_integrate': 10}
        kwargs_numerics_linear = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': False,
                           'max_integrate': 10}
        galkin_linear = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type, anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_linear)

        sigma_v = galkin_linear.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture)
        galkin_log = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type,
                        anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics_log)
        sigma_v2 = galkin_log.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture)
        print(sigma_v, sigma_v2, 'sigma_v linear, sigma_v log')
        print((sigma_v/sigma_v2)**2)

        npt.assert_almost_equal(sigma_v/sigma_v2, 1, decimal=1)

    def test_compare_power_law(self):
        """
        compare power-law profiles analytical vs. numerical
        :return:
        """
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.5
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 1.
        width = 0.3
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0}

        psf_fwhm = 1.  # Gaussian FWHM psf
        kwargs_cosmo = {'D_d': 1000, 'D_s': 1500, 'D_ds': 800}
        kwargs_numerics = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': True,
                           'max_integrate': 100}
        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type, anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        sigma_v = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture)

        kwargs_numerics = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': False,
                           'max_integrate': 10}
        galkin = Galkin(mass_profile_list, light_profile_list, aperture_type=aperture_type, anisotropy_model=anisotropy_type, fwhm=psf_fwhm, kwargs_cosmo=kwargs_cosmo, **kwargs_numerics)
        sigma_v_lin = galkin.vel_disp(kwargs_profile, kwargs_light, kwargs_anisotropy, kwargs_aperture)

        los_disp = AnalyticKinematics(**kwargs_cosmo)
        sigma_v2 = los_disp.vel_disp(gamma, theta_E, r_eff / 0.551, r_ani=r_ani, R_slit=length, dR_slit=width,
                                     FWHM=psf_fwhm, rendering_number=1000)
        print(sigma_v, sigma_v_lin, sigma_v2, 'sigma_v Galkin (log and linear), sigma_v los dispersion')
        npt.assert_almost_equal(sigma_v2/sigma_v, 1, decimal=2)

    def test_projected_light_integral_hernquist(self):
        """

        :return:
        """
        light_profile_list = ['HERNQUIST']
        r_eff = 1.
        kwargs_light = [{'Rs': r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 2
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        npt.assert_almost_equal(light2d, out[0]*2, decimal=3)

    def test_projected_light_integral_hernquist_ellipse(self):
        """

        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE']
        r_eff = 1.
        phi, q = 1, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_light = [{'Rs': r_eff, 'amp': 1.,'e1': e1, 'e2': e2}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 2
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 10)
        npt.assert_almost_equal(light2d, out[0]*2, decimal=3)

    def test_projected_light_integral_pjaffe(self):
        """

        :return:
        """
        light_profile_list = ['PJAFFE']
        kwargs_light = [{'Rs': .5, 'Ra': 0.01, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print(out, 'out')
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic_0(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST']
        kwargs_light = [{'Rs': 0.10535462602138289, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'amp': 3.7114695634960109}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print(out, 'out')
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic_1(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE']
        phi, q = 0.74260706384506325, 0.46728323131925864
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_light = [{'Rs': 0.10535462602138289, 'e1': e1, 'e2': e2, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'amp': 3.7114695634960109}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print(out, 'out')
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_realistic(self):
        """
        realistic test example
        :return:
        """
        light_profile_list = ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']
        phi, q = 0.74260706384506325, 0.46728323131925864
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = -0.33379268413794494, 0.66582356813012267
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)
        kwargs_light = [{'Rs': 0.10535462602138289, 'e1': e1, 'e2': e2, 'center_x': -0.02678473951679429, 'center_y': 0.88691126347462712, 'amp': 3.7114695634960109},
                        {'Rs': 0.44955054610388684, 'e1': e12, 'e2': e22, 'center_x': 0.019536801118136753, 'center_y': 0.0218888643537157, 'Ra': 0.0010000053334891974, 'amp': 967.00280526319796}]
        lightProfile = LightProfile(light_profile_list)
        R = 0.01
        light2d = lightProfile.light_2d(R=R, kwargs_list=kwargs_light)
        out = integrate.quad(lambda x: lightProfile.light_3d(np.sqrt(R**2+x**2), kwargs_light), 0, 100)
        print(out, 'out')
        npt.assert_almost_equal(light2d/(out[0]*2), 1., decimal=3)

    def test_interpolated_sersic(self):
        from lenstronomy.Analysis.lens_analysis import LensAnalysis
        kwargs_light = [{'n_sersic': 2, 'R_sersic': 0.5, 'amp': 1, 'center_x': 0.01, 'center_y': 0.01}]
        kwargs_lens = [{'n_sersic': 2, 'R_sersic': 0.5, 'k_eff': 1, 'center_x': 0.01, 'center_y': 0.01}]
        deltaPix = 0.1
        numPix = 100

        kwargs_interp = LensAnalysis.light2mass_interpol(['SERSIC'], kwargs_lens_light=kwargs_light, numPix=numPix,
                                                                                          deltaPix=deltaPix, subgrid_res=5)
        kwargs_lens_interp = [kwargs_interp]
        from lenstronomy.Analysis.lens_properties import LensProp
        z_lens = 0.5
        z_source = 1.5
        r_ani = 0.62
        kwargs_anisotropy = {'r_ani': r_ani}
        R_slit = 3.8
        dR_slit = 1.
        kwargs_aperture = {'center_ra': 0, 'width': dR_slit, 'length': R_slit, 'angle': 0, 'center_dec': 0}
        aperture_type = 'slit'
        psf_fwhm = 0.7
        anisotropy_model = 'OsipkovMerritt'
        r_eff = 0.5
        kwargs_options = {'lens_model_list': ['SERSIC'],
                          'lens_light_model_list': ['SERSIC']}
        lensProp = LensProp(z_lens, z_source, kwargs_options)

        v_sigma = lensProp.velocity_dispersion_numerical(kwargs_lens, kwargs_light, kwargs_anisotropy,
                                                         kwargs_aperture, psf_fwhm, aperture_type, anisotropy_model,
                                                         MGE_light=True, MGE_mass=True, r_eff=r_eff)
        kwargs_options_interp = {'lens_model_list': ['INTERPOL'],
                                 'lens_light_model_list': ['SERSIC']}
        lensProp_interp = LensProp(z_lens, z_source, kwargs_options_interp)
        v_sigma_interp = lensProp_interp.velocity_dispersion_numerical(kwargs_lens_interp, kwargs_light, kwargs_anisotropy,
                                                         kwargs_aperture, psf_fwhm, aperture_type, anisotropy_model,
                                                         kwargs_numerics={}, MGE_light=True, MGE_mass=True, r_eff=r_eff)
        npt.assert_almost_equal(v_sigma / v_sigma_interp, 1, 1)
        # use as kinematic constraints
        # compare with MGE Sersic kinematic estimate


if __name__ == '__main__':
    pytest.main()

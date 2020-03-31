"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import lenstronomy.Util.multi_gauss_expansion as mge
import numpy as np
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.GalKin.galkin import Galkin


class TestGalkin(object):

    def setup(self):
        pass

    def test_mge_hernquist_light(self):
        """
        compare power-law profiles analytical vs. numerical
        :return:
        """
        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0, 'aperture_type': aperture_type}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}

        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.8
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec

        # mge of light profile
        lightModel = LightModel(light_profile_list)
        r_array = np.logspace(-2, 2, 100)
        flux_r = lightModel.surface_brightness(r_array, 0, kwargs_light)
        amps, sigmas, norm = mge.mge_1d(r_array, flux_r, N=20)
        light_profile_list_mge = ['MULTI_GAUSSIAN']
        kwargs_light_mge = [{'amp': amps, 'sigma': sigmas}]
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}

        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        kwargs_numerics = {'interpol_grid_num': 100, 'log_integration': True,
                           'max_integrate': 100, 'min_integrate': 0.01}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v = galkin.dispersion(kwargs_profile, kwargs_light, kwargs_anisotropy)

        kwargs_model_mge = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list_mge,
                        'anisotropy_model': anisotropy_type}
        galkin = Galkin(kwargs_model=kwargs_model_mge, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v2 = galkin.dispersion(kwargs_profile, kwargs_light_mge, kwargs_anisotropy)

        print(sigma_v, sigma_v2, 'sigma_v Galkin, sigma_v MGEn')
        print((sigma_v/sigma_v2)**2)

        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=2)

    def test_mge_power_law_lens(self):
        """
        compare power-law profiles analytical vs. numerical
        :return:
        """
        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0, 'aperture_type': aperture_type}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}

        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.8
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec

        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # mge of lens profile
        lensModel = LensModel(mass_profile_list)
        r_array = np.logspace(-2, 2, 100)*theta_E
        kappa_r = lensModel.kappa(r_array, 0, kwargs_profile)
        amps, sigmas, norm = mge.mge_1d(r_array, kappa_r, N=20)
        mass_profile_list_mge = ['MULTI_GAUSSIAN_KAPPA']
        kwargs_profile_mge = [{'amp': amps, 'sigma': sigmas}]
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        kwargs_numerics = {'interpol_grid_num': 100, 'log_integration': True,
                           'max_integrate': 100, 'min_integrate': 0.01}

        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v = galkin.dispersion(kwargs_profile, kwargs_light, kwargs_anisotropy)

        kwargs_model = {'mass_profile_list': mass_profile_list_mge,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v2 = galkin.dispersion(kwargs_profile_mge, kwargs_light, kwargs_anisotropy)

        print(sigma_v, sigma_v2, 'sigma_v Galkin, sigma_v MGEn')
        print((sigma_v/sigma_v2)**2)

        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=2)

    def test_mge_light_and_mass(self):
        # anisotropy profile
        anisotropy_model = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0, 'aperture_type': aperture_type}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}

        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.8
        kwargs_light = [{'Rs':  r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec

        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # mge of light profile
        lightModel = LightModel(light_profile_list)
        r_array = np.logspace(-2, 2, 200) * r_eff * 2
        flux_r = lightModel.surface_brightness(r_array, 0, kwargs_light)
        amps, sigmas, norm = mge.mge_1d(r_array, flux_r, N=20)
        light_profile_list_mge = ['MULTI_GAUSSIAN']
        kwargs_light_mge = [{'amp': amps, 'sigma': sigmas}]

        # mge of lens profile
        lensModel = LensModel(mass_profile_list)
        r_array = np.logspace(-2, 2, 200)
        kappa_r = lensModel.kappa(r_array, 0, kwargs_profile)
        amps, sigmas, norm = mge.mge_1d(r_array, kappa_r, N=20)
        mass_profile_list_mge = ['MULTI_GAUSSIAN_KAPPA']
        kwargs_profile_mge = [{'amp': amps, 'sigma': sigmas}]
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_model}
        kwargs_numerics = {'interpol_grid_num': 100, 'log_integration': True,
                           'max_integrate': 100, 'min_integrate': 0.01}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v = galkin.dispersion(kwargs_profile, kwargs_light, kwargs_anisotropy)

        kwargs_model_mge = {'mass_profile_list': mass_profile_list_mge,
                        'light_profile_list': light_profile_list_mge,
                        'anisotropy_model': anisotropy_model}

        galkin = Galkin(kwargs_model=kwargs_model_mge, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics=kwargs_numerics)
        sigma_v2 = galkin.dispersion(kwargs_profile_mge, kwargs_light_mge, kwargs_anisotropy)

        print(sigma_v, sigma_v2, 'sigma_v Galkin, sigma_v MGEn')
        print((sigma_v/sigma_v2)**2)
        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=2)

    def test_sersic_vs_hernquist_kinematics(self):
        """
        attention: this test only works for Sersic indices > \approx 2!
        Lower n_sersic will result in different predictions with the Hernquist assumptions
        replacing the correct Light model!
        :return:
        """
        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'slit'
        length = 3.8
        width = 0.9
        kwargs_aperture = {'length': length, 'width': width, 'center_ra': 0, 'center_dec': 0, 'angle': 0, 'aperture_type': aperture_type}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}

        # light profile
        light_profile_list = ['SERSIC']
        r_sersic = .3
        n_sersic = 2.8
        kwargs_light = [{'amp': 1., 'R_sersic':  r_sersic, 'n_sersic': n_sersic, 'center_x': 0, 'center_y': 0}]  # effective half light radius (2d projected) in arcsec

        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # Hernquist fit to Sersic profile
        profile_analysis = LightProfileAnalysis(LightModel(['SERSIC']))
        r_eff = profile_analysis.half_light_radius(kwargs_light, grid_spacing=0.1, grid_num=100)
        print(r_eff)
        light_profile_list_hernquist = ['HERNQUIST']
        kwargs_light_hernquist = [{'Rs': r_eff*0.551, 'amp': 1.}]

        # mge of light profile
        lightModel = LightModel(light_profile_list)
        r_array = np.logspace(-3, 2, 100) * r_eff * 2
        print(r_sersic/r_eff, 'r_sersic/r_eff')
        flux_r = lightModel.surface_brightness(r_array, 0, kwargs_light)
        amps, sigmas, norm = mge.mge_1d(r_array, flux_r, N=20)
        light_profile_list_mge = ['MULTI_GAUSSIAN']
        kwargs_light_mge = [{'amp': amps, 'sigma': sigmas}]
        print(amps, sigmas, 'amp', 'sigma')
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list_hernquist,
                        'anisotropy_model': anisotropy_type}

        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics={})
        sigma_v = galkin.dispersion(kwargs_profile, kwargs_light_hernquist, kwargs_anisotropy)

        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list_mge,
                        'anisotropy_model': anisotropy_type}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo,
                        kwargs_aperture=kwargs_aperture, kwargs_numerics={})
        sigma_v2 = galkin.dispersion(kwargs_profile, kwargs_light_mge, kwargs_anisotropy)

        print(sigma_v, sigma_v2, 'sigma_v Galkin, sigma_v MGEn')
        print((sigma_v/sigma_v2)**2)

        npt.assert_almost_equal((sigma_v-sigma_v2)/sigma_v2, 0, decimal=1)


if __name__ == '__main__':
    pytest.main()

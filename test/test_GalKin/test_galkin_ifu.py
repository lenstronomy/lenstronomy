"""
Tests for `galkin` module.
"""
import pytest
import numpy.testing as npt
import numpy as np
from lenstronomy.GalKin.galkin_ifu import GalkinIFU
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.analytic_kinematics import AnalyticKinematics



class TestGalkinIFU(object):

    def setup(self):
        np.random.seed(42)

    def test_dispersion_map(self):
        """
        tests whether the old and new version provide the same answer
        """
        # light profile
        light_profile_list = ['HERNQUIST']
        r_eff = 1.5
        kwargs_light = [{'Rs': r_eff, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_mass = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OsipkovMerritt'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as shell
        #aperture_type = 'shell'
        #kwargs_aperture_inner = {'r_in': 0., 'r_out': 0.2, 'center_dec': 0, 'center_ra': 0}

        #kwargs_aperture_outer = {'r_in': 0., 'r_out': 1.5, 'center_dec': 0, 'center_ra': 0}

        # aperture as slit
        r_bins = np.linspace(0, 2, 3)
        kwargs_ifu = {'r_bins': r_bins, 'center_ra': 0, 'center_dec': 0, 'aperture_type': 'IFU_shells'}
        kwargs_aperture = {'aperture_type': 'shell', 'r_in': r_bins[0], 'r_out': r_bins[1], 'center_ra': 0,
                           'center_dec': 0}

        psf_fwhm = 1.  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics = {'sampling_number': 1000, 'interpol_grid_num': 500, 'log_integration': True,
                           'max_integrate': 100}
        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}

        galkinIFU = GalkinIFU(kwargs_ifu, kwargs_psf, kwargs_cosmo, kwargs_model, kwargs_numerics=kwargs_numerics,
                              analytic_kinematics=False)

        sigma_v_ifu = galkinIFU.dispersion_map(kwargs_mass, kwargs_light, kwargs_anisotropy, num_kin_sampling=1000,
                                               num_psf_sampling=100)
        galkin = Galkin(kwargs_model, kwargs_aperture, kwargs_psf, kwargs_cosmo, kwargs_numerics)
        sigma_v = galkin.vel_disp(kwargs_mass, kwargs_light, kwargs_anisotropy)
        npt.assert_almost_equal(sigma_v, sigma_v_ifu[0], decimal=-1)

        galkinIFU = GalkinIFU(kwargs_ifu, kwargs_psf, kwargs_cosmo, kwargs_model, kwargs_numerics=kwargs_numerics,
                              analytic_kinematics=True)
        sigma_v_ifu = galkinIFU.dispersion_map(kwargs_mass={'theta_E': theta_E, 'gamma': gamma}, kwargs_light={'r_eff': r_eff},
                                               kwargs_anisotropy=kwargs_anisotropy, num_kin_sampling=1000,
                                               num_psf_sampling=100)

        galkin_analytic = AnalyticKinematics(kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf, kwargs_cosmo=kwargs_cosmo)
        sigma_v2 = galkin_analytic.vel_disp(gamma, theta_E, r_eff, r_ani=r_ani, rendering_number=2000)
        npt.assert_almost_equal(sigma_v2, sigma_v_ifu[0], decimal=-1)


if __name__ == '__main__':
    pytest.main()

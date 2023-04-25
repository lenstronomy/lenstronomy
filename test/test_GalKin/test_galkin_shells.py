from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.GalKin.galkin_shells import GalkinShells
import numpy as np
import numpy.testing as npt
import pytest
import unittest

class TestGalkinShells(object):

    def test_vel_disp(self):

        # light profile
        light_profile_list = ['HERNQUIST']
        Rs = .5
        kwargs_light = [{'Rs': Rs, 'amp': 1.}]  # effective half light radius (2d projected) in arcsec
        # 0.551 *
        # mass profile
        mass_profile_list = ['SPP']
        theta_E = 1.2
        gamma = 2.
        kwargs_profile = [{'theta_E': theta_E, 'gamma': gamma}]  # Einstein radius (arcsec) and power-law slope

        # anisotropy profile
        anisotropy_type = 'OM'
        r_ani = 2.
        kwargs_anisotropy = {'r_ani': r_ani}  # anisotropy radius [arcsec]

        # aperture as slit
        aperture_type = 'IFU_shells'
        r_bins = np.linspace(start=0, stop=2, num=5)
        kwargs_aperture = {'aperture_type': aperture_type, 'r_bins': r_bins, 'center_ra': 0, 'center_dec': 0}

        psf_fwhm = 0.7  # Gaussian FWHM psf
        kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
        kwargs_numerics_log = {'interpol_grid_num': 1000, 'log_integration': True,
                               'max_integrate': 10, 'min_integrate': 0.001,
                               'lum_weight_int_method': True}

        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': psf_fwhm}
        kwargs_model = {'mass_profile_list': mass_profile_list,
                        'light_profile_list': light_profile_list,
                        'anisotropy_model': anisotropy_type}
        galkin = Galkin(kwargs_model=kwargs_model, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                        kwargs_cosmo=kwargs_cosmo, kwargs_numerics=kwargs_numerics_log)
        galkin_shells = GalkinShells(kwargs_model=kwargs_model, kwargs_aperture=kwargs_aperture, kwargs_psf=kwargs_psf,
                        kwargs_cosmo=kwargs_cosmo, kwargs_numerics=kwargs_numerics_log)

        vel_disp_bins = galkin_shells.dispersion_map(kwargs_mass=kwargs_profile, kwargs_light=kwargs_light,
                                                     kwargs_anisotropy=kwargs_anisotropy)
        disp_map = galkin.dispersion_map(kwargs_mass=kwargs_profile, kwargs_light=kwargs_light,
                                         kwargs_anisotropy=kwargs_anisotropy,
                                         num_kin_sampling=1000, num_psf_sampling=100)
        npt.assert_almost_equal(vel_disp_bins / disp_map, 1, decimal=2)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            kwargs_model = {'mass_profile_list': ['SPP'],
                            'light_profile_list': ['HERNQUIST'],
                            'anisotropy_model': 'const'}
            kwargs_aperture = {'center_ra': 0, 'width': 1, 'length': 1, 'angle': 0, 'center_dec': 0,
                               'aperture_type': 'slit'}
            kwargs_cosmo = {'d_d': 1000, 'd_s': 1500, 'd_ds': 800}
            kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 1}
            GalkinShells(kwargs_model, kwargs_aperture, kwargs_psf, kwargs_cosmo)


if __name__ == '__main__':
    pytest.main()

import pytest
from lenstronomy.SimulationAPI.multiband import MultiBand
import lenstronomy.Util.param_util as param_util
import numpy as np


class TestMultiband(object):

    def setup(self):
        MB = MultiBand()
        # telescope and instruments specifics
        collector_area_DES = 4.  # area of the mirror of a DES-like telescope (in meter^2)
        numPix_DES = 50  # cutout size in number of pixels (numPix x numPix)
        deltaPix_DES = 0.263  # pixel size (in arc seconds)
        readout_noise_DES = 10  # read out noise rms in units of photon counts
        psf_type_DES = 'GAUSSIAN'  # functional (or pixelized) form of PSF

        # exposure specifics
        name = "DES_g_band"
        sky_brightness = 100  # surface brightness of the sky per arc second for a 1 m^2 equivalent collector
        extinction = 1.  # extinction factor (<=1) with =1 no extinction
        exposure_time = 90  # exposure time in units of seconds
        fwhm = 0.1  # full width at half maximum of the PSF

        MB.add_band(name, collector_area_DES, numPix_DES, deltaPix_DES, readout_noise_DES, sky_brightness, extinction,
                    exposure_time, psf_type_DES, fwhm)

        MB.del_band(name=name)
        # exposure specifics
        name = "DES_r_band"
        sky_brightness = 100  # surface brightness of the sky per arc second for a 1 m^2 equivalent collector
        extinction = 1.  # extinction factor (<=1) with =1 no extinction
        exposure_time = 90  # exposure time in units of seconds
        fwhm = 0.7  # full width at half maximum of the PSF

        MB.add_band(name, collector_area_DES, numPix_DES, deltaPix_DES, readout_noise_DES, sky_brightness, extinction,
                    exposure_time, psf_type_DES, fwhm)

        self.MB = MB
        # model specifics

        # list of lens models, supports:

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        # 'SPEMD': Smoothed power-law ellipsoidal mass distribution
        e1, e2 = param_util.phi_q2_ellipticity(0.2, 0.8)
        kwargs_spep = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}
        # 'NONE': no lens
        kwargs_none = {}

        lens_model_list = ['SPEP', 'SHEAR']
        kwargs_lens_list = [kwargs_spep, kwargs_shear]

        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1000., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 10., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        kwargs_lens_light_list = [kwargs_sersic]
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_list = [kwargs_sersic_ellipse]

        kwargs_else = {'sourcePos_x': 0.0, 'sourcePos_y': 0.0,
                       'quasar_amp': 100.}  # quasar point source position in the source plane and intrinsic brightness

        kwargs_options = {'lens_model_list': lens_model_list,
                          'lens_light_model_list': lens_light_model_list,
                          'source_light_model_list': source_model_list,
                          'point_source': True
                          # if True, simulates point source at source position of 'sourcePos_xy' in kwargs_else
                          }
        source_colour = [1., 0.01]
        lens_colour = [0.1, 1]
        quasar_colour = [1, 1]
        self.image_list = self.MB.simulate_bands(kwargs_options, kwargs_lens_list, kwargs_source_list, kwargs_lens_light_list,
                                       kwargs_else, lens_colour, source_colour, quasar_colour, no_noise=False,
                                       source_add=True, lens_light_add=True, point_source_add=True)
        numPix = 10
        deltaPix = 0.1
        self.source_list = self.MB.source_plane(kwargs_options, kwargs_source_list, source_colour, numPix, deltaPix)

    def test_add_remove_band(self):
        assert self.MB.image_name(0) == 'DES_r_band'

    def simulate(self):
        assert len(self.image_list) == 1

    def test_source_plane(self):
        assert len(self.source_list) == 1


if __name__ == '__main__':
    pytest.main()
import unittest
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util
import os
import astropy.io.fits as pyfits


class TestRoman(unittest.TestCase):

    def setUp(self):
        self.F062 = Roman()  # default is F062
        self.F087 = Roman(band='F087', survey_mode='microlensing')
        self.F106 = Roman(band='F106', psf_type='GAUSSIAN')
        self.F129 = Roman(band='F129', psf_type='GAUSSIAN')
        self.F158 = Roman(band='F158', psf_type='GAUSSIAN')
        self.F184 = Roman(band='F184', psf_type='GAUSSIAN')
        self.F146 = Roman(band='F146', survey_mode='microlensing', psf_type='GAUSSIAN')

        kwargs_F062 = self.F062.kwargs_single_band()
        kwargs_F087 = self.F087.kwargs_single_band()
        kwargs_F106 = self.F106.kwargs_single_band()
        kwargs_F129 = self.F129.kwargs_single_band()
        kwargs_F158 = self.F158.kwargs_single_band()
        kwargs_F184 = self.F184.kwargs_single_band()
        kwargs_F146 = self.F146.kwargs_single_band()

        self.F062_band = SingleBand(**kwargs_F062)
        self.F087_band = SingleBand(**kwargs_F087)
        self.F106_band = SingleBand(**kwargs_F106)
        self.F129_band = SingleBand(**kwargs_F129)
        self.F158_band = SingleBand(**kwargs_F158)
        self.F184_band = SingleBand(**kwargs_F184)
        self.F146_band = SingleBand(**kwargs_F146)

        # dictionaries mapping Roman kwargs to SingleBand kwargs
        self.camera_settings = {'read_noise': '_read_noise',
                                'pixel_scale': 'pixel_scale',
                                'ccd_gain': 'ccd_gain'}
        self.obs_settings = {'exposure_time': '_exposure_time',
                             'sky_brightness': '_sky_brightness_',
                             'magnitude_zero_point': '_magnitude_zero_point',
                             'num_exposures': '_num_exposures',
                             'seeing': '_seeing',
                             'psf_type': '_psf_type'}

        self.instrument = Instrument(**self.F062.camera)

    def test_Roman_class(self):
        default = self.F062
        explicit_F062 = Roman(band='F062')
        self.assertEqual(explicit_F062.camera, default.camera)
        self.assertEqual(explicit_F062.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = Roman(band='g')
        
        with self.assertRaises(ValueError):
            bad_band_2 = Roman(band='9')

        with self.assertRaises(ValueError):
            bad_psf = Roman(psf_type='blah')
        
        with self.assertRaises(ValueError):
            bad_band_wide = Roman(band='F087')

        with self.assertRaises(ValueError):
            bad_band_microlensing = Roman(band='F062', survey_mode='microlensing')

        with self.assertRaises(ValueError):
            bad_survey_mode = Roman(survey_mode='blah')

    def test_Roman_camera(self):
        # comparing camera settings in Roman instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(self.F062.camera[config], getattr(self.instrument, setting), msg=f"{config} did not match")

    def test_Roman_obs(self):
        # comparing obs settings in HST instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(self.F062.obs[config], getattr(self.F062_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F087.obs[config], getattr(self.F087_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F106.obs[config], getattr(self.F106_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F129.obs[config], getattr(self.F129_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F158.obs[config], getattr(self.F158_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F184.obs[config], getattr(self.F184_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.F146.obs[config], getattr(self.F146_band, setting), msg=f"{config} did not match")

    def test_Roman_psf_pixel(self):
        self.F062_pixel = Roman(psf_type = 'PIXEL')
        import lenstronomy
        module_path = os.path.dirname(lenstronomy.__file__)
        psf_filename = os.path.join(module_path, 'SimulationAPI\ObservationConfig\PSF_models\F062.fits')
        kernel = pyfits.getdata(psf_filename)
        self.assertEqual(self.F062_pixel.obs['kernel_point_source'].all(), kernel.all(), msg="PSF did not match")

    def test_kwargs_single_band(self):
        kwargs_F062 = util.merge_dicts(self.F062.camera, self.F062.obs)
        self.assertEqual(self.F062.kwargs_single_band(), kwargs_F062)

if __name__ == '__main__':
    unittest.main()

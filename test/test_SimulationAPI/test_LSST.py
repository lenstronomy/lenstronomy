import unittest
import lenstronomy.SimulationAPI.ObservationConfig.LSST as LSST
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util

class TestLSST(unittest.TestCase):

    def setUp(self):
        self.instrument = Instrument(**LSST.camera)

        kwargs_g_band = util.merge_dicts(LSST.camera, LSST.g_band_obs)
        kwargs_r_band = util.merge_dicts(LSST.camera, LSST.r_band_obs)
        kwargs_i_band = util.merge_dicts(LSST.camera, LSST.i_band_obs)

        self.g_band = SingleBand(**kwargs_g_band)
        self.r_band = SingleBand(**kwargs_r_band)
        self.i_band = SingleBand(**kwargs_i_band)

        # dictionaries mapping LSST kwargs to SingleBand kwargs
        self.camera_settings = {'read_noise': '_read_noise',
               'pixel_scale': 'pixel_scale',
               'ccd_gain': 'ccd_gain'}
        self.obs_settings = {'exposure_time': '_exposure_time',
               'sky_brightness': '_sky_brightness_',
               'magnitude_zero_point': '_magnitude_zero_point',
               'num_exposures': '_num_exposures',
               'seeing': '_seeing',
               'psf_type': '_psf_type' }

    def test_camera(self):
        for config, setting in self.camera_settings.items():
            self.assertEqual(LSST.camera[config], getattr(self.instrument, setting), msg=f"{config} did not match")

    def test_bands(self):
        for config, setting in self.obs_settings.items():
            self.assertEqual(LSST.g_band_obs[config], getattr(self.g_band, setting), msg=f"{config} did not match")
            self.assertEqual(LSST.r_band_obs[config], getattr(self.r_band, setting), msg=f"{config} did not match")
            self.assertEqual(LSST.i_band_obs[config], getattr(self.i_band, setting), msg=f"{config} did not match")

    # test SimApi?

if __name__ == '__main__':
    unittest.main()
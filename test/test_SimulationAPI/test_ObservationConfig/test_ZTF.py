import unittest
from lenstronomy.SimulationAPI.ObservationConfig.ZTF import ZTF
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util


class TestZTF(unittest.TestCase):

    def setUp(self):
        self.g = ZTF()  # default is g_band
        self.r = ZTF(band='r')
        self.i = ZTF(band='i')

        kwargs_g_band = self.g.kwargs_single_band()
        kwargs_r_band = self.r.kwargs_single_band()
        kwargs_i_band = self.i.kwargs_single_band()

        self.g_band = SingleBand(**kwargs_g_band)
        self.r_band = SingleBand(**kwargs_r_band)
        self.i_band = SingleBand(**kwargs_i_band)

        # dictionaries mapping ZTF kwargs to SingleBand kwargs
        self.camera_settings = {'read_noise': '_read_noise',
                                'pixel_scale': 'pixel_scale',
                                'ccd_gain': 'ccd_gain'}
        self.obs_settings = {'exposure_time': '_exposure_time',
                             'sky_brightness': '_sky_brightness_',
                             'magnitude_zero_point': '_magnitude_zero_point',
                             'num_exposures': '_num_exposures',
                             'seeing': '_seeing',
                             'psf_type': '_psf_type'}

        self.instrument = Instrument(**self.g.camera)

    def test_ZTF_class(self):
        default = self.g
        explicit_g = ZTF(band='g')
        self.assertEqual(explicit_g.camera, default.camera)
        self.assertEqual(explicit_g.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = ZTF(band='z')

        with self.assertRaises(ValueError):
            bad_psf = ZTF(psf_type='blah')

        single_year = ZTF(coadd_years=1)
        self.assertEqual(single_year.obs["num_exposures"], 13)
        with self.assertRaises(ValueError):
            bad_coadd_years = ZTF(coadd_years=10)

    def test_ZTF_camera(self):
        # comparing camera settings in ZTF instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(self.g.camera[config], getattr(self.instrument, setting), msg=f"{config} did not match")

    def test_ZTF_obs(self):
        # comparing obs settings in ZTF instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(self.g.obs[config], getattr(self.g_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.r.obs[config], getattr(self.r_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.i.obs[config], getattr(self.i_band, setting), msg=f"{config} did not match")

    def test_kwargs_single_band(self):
        kwargs_g = util.merge_dicts(self.g.camera, self.g.obs)
        self.assertEqual(self.g.kwargs_single_band(), kwargs_g)

if __name__ == '__main__':
    unittest.main()

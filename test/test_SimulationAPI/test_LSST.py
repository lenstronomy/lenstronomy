import unittest
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util

class TestLSST(unittest.TestCase):

    def setUp(self):
        self.g = LSST()  # default is g_band
        self.r = LSST(band='r')
        self.i = LSST(band='i')

        kwargs_g_band = self.g.kwargs_single_band()
        kwargs_r_band = self.r.kwargs_single_band()
        kwargs_i_band = self.i.kwargs_single_band()

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
                             'psf_type': '_psf_type'}

        self.instrument = Instrument(**self.g.camera)

    def test_LSST_class(self):
        default = self.g
        explicit_g = LSST(band='g')
        self.assertEqual(explicit_g.camera, default.camera)
        self.assertEqual(explicit_g.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = LSST(band='u')

        with self.assertRaises(ValueError):
            bad_psf = LSST(psf_type='blah')

        with self.assertRaises(ValueError):
            bad_coadd_years = LSST(psf_type=100)

    def test_LSST_camera(self):
        # comparing camera settings in LSST instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(self.g.camera[config], getattr(self.instrument, setting), msg=f"{config} did not match")

    def test_LSST_obs(self):
        # comparing obs settings in LSST instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(self.g.obs[config], getattr(self.g_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.r.obs[config], getattr(self.r_band, setting), msg=f"{config} did not match")
            self.assertEqual(self.i.obs[config], getattr(self.i_band, setting), msg=f"{config} did not match")

    def test_kwargs_single_band(self):
        kwargs_g = util.merge_dicts(self.g.camera, self.g.obs)
        self.assertEqual(self.g.kwargs_single_band(), kwargs_g)

if __name__ == '__main__':
    unittest.main()

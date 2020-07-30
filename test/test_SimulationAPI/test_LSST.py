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

    def test_camera(self):
        assert self.instrument.ccd_gain == LSST.camera['ccd_gain']
        assert self.instrument.pixel_scale == LSST.camera['pixel_scale']
        assert self.instrument._read_noise == LSST.camera['read_noise']

    def test_g_band(self):
        assert self.g_band.ccd_gain == LSST.camera['ccd_gain']
        assert self.g_band.pixel_scale == LSST.camera['pixel_scale']
        assert self.g_band._read_noise == LSST.camera['read_noise']

        assert self.g_band._exposure_time == LSST.g_band_obs['exposure_time']
        assert self.g_band._sky_brightness_ == LSST.g_band_obs['sky_brightness']
        assert self.g_band._num_exposures == LSST.g_band_obs['num_exposures']
        assert self.g_band._seeing == LSST.g_band_obs['seeing']
        assert self.g_band._psf_type == LSST.g_band_obs['psf_type']
        assert self.g_band._magnitude_zero_point == LSST.g_band_obs['magnitude_zero_point']

    # add r and i band tests?
    # test SimApi?

if __name__ == '__main__':
    unittest.main()
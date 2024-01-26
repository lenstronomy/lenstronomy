import unittest
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util


class TestHST(unittest.TestCase):
    def setup_method(self):
        self.TDLMC_F160W = HST()  # default is TDLMC_F160W
        self.F160W = HST(band="F160W")
        self.F160W2 = HST(band="F160W", psf_type="GAUSSIAN")

        kwargs_TDLMC_F160W = self.TDLMC_F160W.kwargs_single_band()
        kwargs_F160W = self.F160W.kwargs_single_band()
        kwargs_F160W2 = self.F160W2.kwargs_single_band()

        self.TDLMC_F160W_band = SingleBand(**kwargs_TDLMC_F160W)
        self.F160W_band = SingleBand(**kwargs_F160W)
        self.F160W2_band = SingleBand(**kwargs_F160W2)

        # dictionaries mapping HST kwargs to SingleBand kwargs
        self.camera_settings = {
            "read_noise": "_read_noise",
            "pixel_scale": "pixel_scale",
            "ccd_gain": "ccd_gain",
        }
        self.obs_settings = {
            "exposure_time": "_exposure_time",
            "sky_brightness": "_sky_brightness_",
            "magnitude_zero_point": "_magnitude_zero_point",
            "num_exposures": "_num_exposures",
            "seeing": "_seeing",
            "psf_type": "_psf_type",
        }

        self.instrument = Instrument(**self.TDLMC_F160W.camera)

    def test_HST_class(self):  # TODO: update; also text pixel/gaussian
        default = self.TDLMC_F160W
        explicit_TDLMC_F160W = HST(band="TDLMC_F160W", psf_type="PIXEL")
        self.assertEqual(explicit_TDLMC_F160W.camera, default.camera)
        self.assertEqual(explicit_TDLMC_F160W.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = HST(band="g")

        with self.assertRaises(ValueError):
            bad_psf = HST(psf_type="blah")

        with self.assertRaises(ValueError):
            bad_coadd_years = HST(coadd_years=100)

    def test_HST_camera(self):
        # comparing camera settings in HST instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(
                self.TDLMC_F160W.camera[config],
                getattr(self.instrument, setting),
                msg=f"{config} did not match",
            )

    def test_HST_obs(self):
        # comparing obs settings in HST instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(
                self.TDLMC_F160W.obs[config],
                getattr(self.TDLMC_F160W_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.F160W.obs[config],
                getattr(self.F160W_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.F160W2.obs[config],
                getattr(self.F160W2_band, setting),
                msg=f"{config} did not match",
            )

    def test_kwargs_single_band(self):
        kwargs_TDLMC_F160W = util.merge_dicts(
            self.TDLMC_F160W.camera, self.TDLMC_F160W.obs
        )
        self.assertEqual(self.TDLMC_F160W.kwargs_single_band(), kwargs_TDLMC_F160W)


if __name__ == "__main__":
    unittest.main()

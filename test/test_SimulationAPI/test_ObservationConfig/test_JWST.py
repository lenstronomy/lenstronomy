import unittest
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util


class TestJWST(unittest.TestCase):
    def setUp(self):
        self.F200W = JWST()  # default is F200W
        self.F356W = JWST(band="F356W")
        self.F356W2 = JWST(band="F356W", psf_type="GAUSSIAN")

        kwargs_F200W = self.F200W.kwargs_single_band()
        kwargs_F356W = self.F356W.kwargs_single_band()
        kwargs_F356W2 = self.F356W2.kwargs_single_band()

        self.F200W_band = SingleBand(**kwargs_F200W)
        self.F356W_band = SingleBand(**kwargs_F356W)
        self.F356W2_band = SingleBand(**kwargs_F356W2)

        # dictionaries mapping JWST kwargs to SingleBand kwargs
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

        self.instrument = Instrument(**self.F200W.camera)

    def test_JWST_class(self):
        default = self.F200W
        explicit_F200W = JWST(band="F200W", psf_type="PIXEL")
        self.assertEqual(explicit_F200W.camera, default.camera)
        self.assertEqual(explicit_F200W.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = JWST(band="g")

        with self.assertRaises(ValueError):
            bad_psf = JWST(psf_type="blah")

        with self.assertRaises(ValueError):
            bad_coadd_years = JWST(coadd_years=100)

    def test_JWST_camera(self):
        # comparing camera settings in JWST instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(
                self.F200W.camera[config],
                getattr(self.instrument, setting),
                msg=f"{config} did not match",
            )

    def test_JWST_obs(self):
        # comparing obs settings in JWST instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(
                self.F200W.obs[config],
                getattr(self.F200W_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.F356W.obs[config],
                getattr(self.F356W_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.F356W2.obs[config],
                getattr(self.F356W2_band, setting),
                msg=f"{config} did not match",
            )

    def test_kwargs_single_band(self):
        kwargs_F200W = util.merge_dicts(self.F200W.camera, self.F200W.obs)
        self.assertEqual(self.F200W.kwargs_single_band(), kwargs_F200W)


if __name__ == "__main__":
    unittest.main()

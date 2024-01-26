import unittest
from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util


class TestDES(unittest.TestCase):
    def setup_method(self):
        self.g = DES()  # default is g_band
        self.r = DES(band="r")
        self.i = DES(band="i")
        self.z = DES(band="z")
        self.Y = DES(band="Y")

        kwargs_g_band = self.g.kwargs_single_band()
        kwargs_r_band = self.r.kwargs_single_band()
        kwargs_i_band = self.i.kwargs_single_band()
        kwargs_z_band = self.z.kwargs_single_band()
        kwargs_Y_band = self.Y.kwargs_single_band()

        self.g_band = SingleBand(**kwargs_g_band)
        self.r_band = SingleBand(**kwargs_r_band)
        self.i_band = SingleBand(**kwargs_i_band)
        self.z_band = SingleBand(**kwargs_z_band)
        self.Y_band = SingleBand(**kwargs_Y_band)

        # dictionaries mapping DES kwargs to SingleBand kwargs
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

        self.instrument = Instrument(**self.g.camera)

    def test_DES_class(self):
        default = self.g
        explicit_g = DES(band="g")
        self.assertEqual(explicit_g.camera, default.camera)
        self.assertEqual(explicit_g.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = DES(band="u")

        with self.assertRaises(ValueError):
            bad_psf = DES(psf_type="blah")

        single_year = DES(coadd_years=1)
        self.assertEqual(single_year.obs["num_exposures"], 3)
        with self.assertRaises(ValueError):
            bad_coadd_years = DES(coadd_years=10)

    def test_DES_camera(self):
        # comparing camera settings in DES instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(
                self.g.camera[config],
                getattr(self.instrument, setting),
                msg=f"{config} did not match",
            )

    def test_DES_obs(self):
        # comparing obs settings in DES instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(
                self.g.obs[config],
                getattr(self.g_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.r.obs[config],
                getattr(self.r_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.i.obs[config],
                getattr(self.i_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.z.obs[config],
                getattr(self.z_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.Y.obs[config],
                getattr(self.Y_band, setting),
                msg=f"{config} did not match",
            )

    def test_kwargs_single_band(self):
        kwargs_g = util.merge_dicts(self.g.camera, self.g.obs)
        self.assertEqual(self.g.kwargs_single_band(), kwargs_g)


if __name__ == "__main__":
    unittest.main()

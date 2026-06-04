import unittest
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.observation_api import Instrument, SingleBand
import lenstronomy.Util.util as util


class TestEuclid(unittest.TestCase):
    def setUp(self):
        self.VIS = Euclid()
        self.Y = Euclid(band="Y")
        self.J = Euclid(band="J")
        self.H = Euclid(band="H")

        kwargs_VIS = self.VIS.kwargs_single_band()
        kwargs_Y = self.Y.kwargs_single_band()
        kwargs_J = self.J.kwargs_single_band()
        kwargs_H = self.H.kwargs_single_band()

        self.VIS_band = SingleBand(**kwargs_VIS)
        self.Y_band = SingleBand(**kwargs_Y)
        self.J_band = SingleBand(**kwargs_J)
        self.H_band = SingleBand(**kwargs_H)

        # dictionaries mapping Euclid kwargs to SingleBand kwargs
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

        self.VIS_instrument = Instrument(**self.VIS.camera)
        self.Y_instrument = Instrument(**self.Y.camera)
        self.J_instrument = Instrument(**self.J.camera)
        self.H_instrument = Instrument(**self.H.camera)

    def test_Euclid_class(self):
        default = self.VIS
        explicit = Euclid(band="VIS")
        self.assertEqual(explicit.camera, default.camera)
        self.assertEqual(explicit.obs, default.obs)

        with self.assertRaises(ValueError):
            bad_band = Euclid(band="g")

        with self.assertRaises(ValueError):
            bad_psf = Euclid(psf_type="pixel")

        single_year = Euclid(coadd_years=2)
        self.assertEqual(single_year.obs["num_exposures"], 1)
        with self.assertRaises(ValueError):
            bad_coadd_years = Euclid(coadd_years=7)

    def test_Euclid_camera(self):
        # comparing camera settings in Euclid instance with those in Instrument instance
        for config, setting in self.camera_settings.items():
            self.assertEqual(
                self.VIS.camera[config],
                getattr(self.VIS_instrument, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.Y.camera[config],
                getattr(self.Y_instrument, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.J.camera[config],
                getattr(self.J_instrument, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.H.camera[config],
                getattr(self.H_instrument, setting),
                msg=f"{config} did not match",
            )

    def test_coadd_years(self):
        for band in ["VIS", "Y", "J", "H"]:
            two_year = Euclid(band=band, coadd_years=2)
            default = Euclid(band=band)

            self.assertEqual(two_year.obs["num_exposures"], 1)
            self.assertEqual(default.obs["num_exposures"], 4)

    def test_Euclid_obs(self):
        # comparing obs settings in Euclid instance with those in SingleBand instance
        for config, setting in self.obs_settings.items():
            self.assertEqual(
                self.VIS.obs[config],
                getattr(self.VIS_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.Y.obs[config],
                getattr(self.Y_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.J.obs[config],
                getattr(self.J_band, setting),
                msg=f"{config} did not match",
            )
            self.assertEqual(
                self.H.obs[config],
                getattr(self.H_band, setting),
                msg=f"{config} did not match",
            )

    def test_kwargs_single_band(self):
        kwargs_VIS = util.merge_dicts(self.VIS.camera, self.VIS.obs)
        self.assertEqual(self.VIS.kwargs_single_band(), kwargs_VIS)


if __name__ == "__main__":
    unittest.main()

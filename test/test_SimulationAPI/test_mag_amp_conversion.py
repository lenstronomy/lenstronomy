from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion
import numpy.testing as npt


class TestMagAmpConversion(object):
    """"""

    def setup_method(self):
        kwargs_model = {
            "lens_light_model_list": ["GAUSSIAN"],
            "source_light_model_list": ["GAUSSIAN"],
            "point_source_model_list": ["SOURCE_POSITION"],
            "lens_model_list": ["SIS"],
        }
        self.mag_zero_point = 30
        self.mag_amp = MagAmpConversion(
            kwargs_model=kwargs_model, magnitude_zero_point=self.mag_zero_point
        )

    def test_magnitude2amplitude(self):
        # input with nones
        kwargs_lens_light, kwargs_source, kwargs_ps = self.mag_amp.magnitude2amplitude(
            kwargs_lens_light_mag=None, kwargs_source_mag=None, kwargs_ps_mag=None
        )
        assert kwargs_lens_light is None
        assert kwargs_source is None
        assert kwargs_ps is None

        kwargs_lens_light = [
            {"magnitude": self.mag_zero_point, "sigma": 1, "center_x": 0, "center_y": 0}
        ]
        kwargs_source = [
            {
                "magnitude": self.mag_zero_point - 1,
                "sigma": 1,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        kwargs_ps = [
            {"magnitude": self.mag_zero_point, "ra_source": 0, "dec_source": 0}
        ]

        kwargs_lens_light_amp, kwargs_source_amp, kwargs_ps_amp = (
            self.mag_amp.magnitude2amplitude(
                kwargs_lens_light_mag=kwargs_lens_light,
                kwargs_source_mag=kwargs_source,
                kwargs_ps_mag=kwargs_ps,
            )
        )
        amp = kwargs_lens_light_amp[0]["amp"]
        npt.assert_almost_equal(amp, 1)
        amp = kwargs_source_amp[0]["amp"]
        npt.assert_almost_equal(amp, 2.5, decimal=1)

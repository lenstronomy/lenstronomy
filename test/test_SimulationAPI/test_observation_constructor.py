import lenstronomy.SimulationAPI.observation_constructor as constructor
from lenstronomy.SimulationAPI.data_api import SingleBand
import unittest


class TestObservationConstructor(unittest.TestCase):

    def setup(self):
        pass

    def test_constructor(self):
        instrument_name = 'LSST'
        observation_name = 'LSST_g_band'
        kwargs_data = constructor.observation_constructor(instrument_name=instrument_name, observation_name=observation_name)
        data = SingleBand(**kwargs_data)
        assert data.pixel_scale == 0.263
        assert data.exposure_time == 900

        obs_name_list = constructor.observation_name_list
        inst_name_list = constructor.instrument_name_list
        for obs_name in obs_name_list:
            for inst_name in inst_name_list:
                constructor.observation_constructor(instrument_name=inst_name, observation_name=obs_name)
        with self.assertRaises(ValueError):
            kwargs_data = constructor.observation_constructor(instrument_name='wrong', observation_name='LSST_g_band')
            SingleBand(**kwargs_data)
        with self.assertRaises(ValueError):
            kwargs_data = constructor.observation_constructor(instrument_name='LSST', observation_name='wrong')
            SingleBand(**kwargs_data)
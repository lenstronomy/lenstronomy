import lenstronomy.SimulationAPI.observation_constructor as constructor


class TestObservationConstructor(object):

    def setup(self):
        pass

    def test_constructor(self):
        instrument_name = 'LSST'
        observation_name = 'LSST_g_band'
        data = constructor.observation_constructor(instrument_name=instrument_name, observation_name=observation_name)
        assert data.pixel_scale == 0.263
        assert data.exposure_time == 900
from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.SimulationAPI.observation_constructor as constructor
import pytest


class TestModelAPI(object):

    def setup(self):

        numpix = 10
        instrument_name = 'LSST'
        observation_name = 'LSST_g_band'
        kwargs_single_band = constructor.observation_constructor(instrument_name=instrument_name,
                                                          observation_name=observation_name)
        kwargs_model = {'lens_model_list': ['SIS'], 'z_lens': None, 'z_source': None, 'lens_redshift_list': None,
                        'multi_plane': False, 'source_light_model_list': ['GAUSSIAN'],
                        'lens_light_model_list': ['SERSIC'], 'point_source_model_list':['UNLENSED'],
                        'source_redshift_list': None}
        kwargs_numerics = {'subgrid_res': 2}

        self.api = SimAPI(numpix, kwargs_single_band, kwargs_model, kwargs_numerics)

    def test_image_model_class(self):
        model = self.api.image_model_class
        assert model.LensModel.lens_model_list[0] == 'SIS'


if __name__ == '__main__':
    pytest.main()

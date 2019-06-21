from lenstronomy.SimulationAPI.model_api import ModelAPI
import pytest


class TestModelAPI(object):

    def setup(self):
        self.api = ModelAPI(lens_model_list=['SIS'], z_lens=None, z_source=None, lens_redshift_list=None,
                            multi_plane=False, source_light_model_list=['GAUSSIAN'], lens_light_model_list=['SERSIC'],
                            point_source_model_list=['UNLENSED'], source_redshift_list=None, cosmo=None)

    def test_lens_model_class(self):
        model = self.api.lens_model_class
        assert model.lens_model_list[0] == 'SIS'

    def test_lens_light_model_class(self):
        model = self.api.lens_light_model_class
        assert model.profile_type_list[0] == 'SERSIC'

    def test_source_model_class(self):
        model = self.api.source_model_class
        assert model.profile_type_list[0] == 'GAUSSIAN'

    def test_point_source_model_class(self):
        model = self.api.point_source_model_class
        assert model.point_source_type_list[0] == 'UNLENSED'

    def test_source_position(self):
        api = ModelAPI(lens_model_list=['SIS'], z_lens=None, z_source=None, lens_redshift_list=None,
                            multi_plane=False, source_light_model_list=['GAUSSIAN'], lens_light_model_list=['SERSIC'],
                            point_source_model_list=['SOURCE_POSITION'], source_redshift_list=None, cosmo=None)
        model = api.point_source_model_class
        assert model.point_source_type_list[0] == 'SOURCE_POSITION'


if __name__ == '__main__':
    pytest.main()

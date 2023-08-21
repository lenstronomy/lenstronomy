from lenstronomy.SimulationAPI.model_api import ModelAPI
import pytest
import numpy.testing as npt
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class TestModelAPI(object):

    def setup_method(self):
        self.api = ModelAPI(lens_model_list=['SIS'], z_lens=None, z_source=None, lens_redshift_list=None,
                            source_light_model_list=['GAUSSIAN'], lens_light_model_list=['SERSIC'],
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
                            source_light_model_list=['GAUSSIAN'], lens_light_model_list=['SERSIC'],
                            point_source_model_list=['SOURCE_POSITION'], source_redshift_list=None, cosmo=None)
        model = api.point_source_model_class
        assert model.point_source_type_list[0] == 'SOURCE_POSITION'

    def test_physical2lensing_conversion(self):
        lens_redshift_list = [0.5, 1]
        z_source_convention = 2
        api = ModelAPI(lens_model_list=['SIS', 'NFW'], lens_redshift_list=lens_redshift_list,
                       z_source_convention=z_source_convention, cosmo=None, z_source=z_source_convention)

        kwargs_mass = [{'sigma_v': 200, 'center_x': 0, 'center_y': 0},
                       {'M200': 10**13, 'concentration': 5, 'center_x': 1, 'center_y': 1}]
        kwargs_lens = api.physical2lensing_conversion(kwargs_mass)

        theta_E = kwargs_lens[0]['theta_E']
        lens_cosmo = LensCosmo(z_lens=lens_redshift_list[0], z_source=z_source_convention)
        theta_E_test = lens_cosmo.sis_sigma_v2theta_E(kwargs_mass[0]['sigma_v'])
        npt.assert_almost_equal(theta_E, theta_E_test, decimal=7)

        alpha_Rs = kwargs_lens[1]['alpha_Rs']
        lens_cosmo = LensCosmo(z_lens=lens_redshift_list[1], z_source=z_source_convention)
        Rs_new , alpha_Rs_new = lens_cosmo.nfw_physical2angle(kwargs_mass[1]['M200'], kwargs_mass[1]['concentration'])
        npt.assert_almost_equal(alpha_Rs, alpha_Rs_new, decimal=7)

        api = ModelAPI(lens_model_list=['SIS', 'NFW'], z_lens=0.5, z_source_convention=z_source_convention, cosmo=None,
                       z_source=z_source_convention)
        kwargs_lens = api.physical2lensing_conversion(kwargs_mass)

        theta_E = kwargs_lens[0]['theta_E']
        lens_cosmo = LensCosmo(z_lens=0.5, z_source=z_source_convention)
        theta_E_test = lens_cosmo.sis_sigma_v2theta_E(kwargs_mass[0]['sigma_v'])
        npt.assert_almost_equal(theta_E, theta_E_test, decimal=7)


if __name__ == '__main__':
    pytest.main()

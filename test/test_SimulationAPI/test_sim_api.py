from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.SimulationAPI.observation_constructor as constructor
import pytest
import numpy.testing as npt


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
        kwargs_numerics = {'supersampling_factor': 2}

        self.api = SimAPI(numpix, kwargs_single_band, kwargs_model, kwargs_numerics)

    def test_image_model_class(self):
        model = self.api.image_model_class
        assert model.LensModel.lens_model_list[0] == 'SIS'

    def test_magnitude2amplitude(self):
        kwargs_lens_light_mag = [{'magnitude': 28, 'R_sersic': 1., 'n_sersic': 2, 'center_x': 0, 'center_y': 0}]
        kwargs_source_mag = [{'magnitude': 30, 'sigma_x': 0.3, 'sigma_y': 0.6, 'center_x': 0, 'center_y': 0}]
        kwargs_ps_mag = [{'magnitude': [30], 'ra_image': [0], 'dec_image': [0]}]
        kwargs_lens_light, kwargs_source, kwargs_ps = self.api.magnitude2amplitude(kwargs_lens_light_mag, kwargs_source_mag,
                                                                        kwargs_ps_mag)
        npt.assert_almost_equal(kwargs_lens_light[0]['amp'], 0.38680586575451237, decimal=5)
        npt.assert_almost_equal(kwargs_source[0]['amp'], 1, decimal=5)
        npt.assert_almost_equal(kwargs_ps[0]['point_amp'][0], 1, decimal=5)


if __name__ == '__main__':
    pytest.main()

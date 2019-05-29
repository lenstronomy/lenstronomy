import numpy as np
import numpy.testing as npt
import lenstronomy.SimulationAPI.observation_constructor as constructor
import pytest
import lenstronomy.Util.data_util as data_util


class TestPointSourceVariability(object):

    def setup(self):
        pass

    def test_image(self):
        # we define a time variable function in magnitude space
        def var_func(time):
            sigma = 100
            mag_0 = 30
            cps = np.exp(-time ** 2 / (2 * sigma ** 2))
            mag = data_util.cps2magnitude(cps, magnitude_zero_point=0)
            mag_norm = data_util.cps2magnitude(1, magnitude_zero_point=0)
            mag_return = -mag + mag_norm + mag_0
            return mag_return

        kwargs_model_time_var = {'lens_model_list': ['SPEP', 'SHEAR'],  # list of lens models to be used
                                 'lens_light_model_list': ['SERSIC_ELLIPSE'],
                                 # list of unlensed light models to be used
                                 'source_light_model_list': ['SERSIC_ELLIPSE'],
                                 # list of extended source models to be used
                                 'z_lens': 0.5, 'z_source': 2
                                 }
        instrument_name = 'LSST'
        observation_name = 'LSST_g_band'
        kwargs_single_band = constructor.observation_constructor(instrument_name=instrument_name,
                                                                 observation_name=observation_name)


        kwargs_numerics = {}
        numpix = 20
        # source position
        source_x, source_y = 0.01, 0.1
        # lens light
        kwargs_lens_light_mag_g = [
            {'magnitude': 100, 'R_sersic': .6, 'n_sersic': 4, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0}]
        # source light
        kwargs_source_mag_g = [
            {'magnitude': 100, 'R_sersic': 0.3, 'n_sersic': 1, 'e1': -0.3, 'e2': -0.2, 'center_x': 0, 'center_y': 0}]

        kwargs_lens = [
            {'theta_E': 1, 'gamma': 2, 'e1': 0.1, 'e2': -0.1, 'center_x': 0, 'center_y': 0},  # SIE model
            {'e1': 0.03, 'e2': 0.01}  # SHEAR model
        ]

        from lenstronomy.SimulationAPI.point_source_variability import PointSourceVariability
        ps_var = PointSourceVariability(source_x, source_y, var_func, numpix, kwargs_single_band, kwargs_model_time_var,
                                        kwargs_numerics,
                                        kwargs_lens, kwargs_source_mag_g, kwargs_lens_light_mag_g, kwargs_ps_mag=None)

        time = 0
        image_g = ps_var.image_time(time=time)
        npt.assert_almost_equal(np.sum(image_g), 8, decimal=1)

        t_days = ps_var.delays
        assert len(t_days) == 4


if __name__ == '__main__':
    pytest.main()

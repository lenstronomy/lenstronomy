import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensAnalysis.lens_analysis import LensAnalysis


class TestLensAnalysis(object):

    def setup(self):
        self.kwargs_options = { 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'],
                               'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        self.analysis = LensAnalysis(self.kwargs_options, kwargs_data={})
        self.kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]


    def test_get_magnification_model(self):
        kwargs_else = {'ra_pos': np.array([1., 1., 2.]), 'dec_pos': np.array([-1., 0., 0.])}
        x_pos, y_pos, mag = self.analysis.magnification_model(self.kwargs_lens, kwargs_else)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_profile_slope(self):
        kwargs_options = {'lens_model_list': ['SPP'], 'lens_model_internal_bool': [True]}
        analysis = LensAnalysis(kwargs_options, kwargs_data={})
        gamma_in = 2.
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = analysis.profile_slope(kwargs_lens, kwargs_else={})
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)
        gamma_in = 1.7
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = analysis.profile_slope(kwargs_lens, kwargs_else={})
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        gamma_in = 2.5
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = analysis.profile_slope(kwargs_lens, kwargs_else={})
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)



if __name__ == '__main__':
    pytest.main()
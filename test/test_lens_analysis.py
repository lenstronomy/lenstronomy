import numpy as np
import numpy.testing as npt
import pytest

from lenstronomy.LensAnalysis.lens_analysis import LensAnalysis
from lenstronomy.ImSim.lens_model import LensModel
import astrofunc.util as util
from astrofunc.util import Util_class


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

    def test_critical_curves(self):
        kwargs_options = {'lens_model_list': ['SPEP']}
        deltaPix = 0.05
        numPix = 100
        x_grid, y_grid, x_0, y_0, ra_0, dec_0, Matrix, Matrix_inv = util.make_grid_with_coordtransform(numPix=numPix, deltapix=deltaPix, subgrid_res=1)
        kwargs_data = {
            'deltaPix': deltaPix, 'numPix_xy': (numPix, numPix)
            , 'x_coords': x_grid, 'y_coords': y_grid
            , 'image_data': np.zeros_like(x_grid)
            }
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'q': 0.8, 'phi_G': 1., 'center_x': 0, 'center_y': 0}]
        analysis = LensAnalysis(kwargs_options, kwargs_data)
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = analysis.critical_curve(kwargs_lens, kwargs_else={})

        import matplotlib.pyplot as plt
        util_class = Util_class()
        lensModel = LensModel(kwargs_options)
        x_grid_high_res, y_grid_high_res = util_class.make_subgrid(kwargs_data['x_coords'], kwargs_data['y_coords'], 10)
        mag_high_res = util.array2image(
            lensModel.magnification(x_grid_high_res, y_grid_high_res, kwargs_lens, kwargs_else={}))

        cs = plt.contour(util.array2image(x_grid_high_res), util.array2image(y_grid_high_res), mag_high_res, [0],
                        alpha=0.0)
        paths = cs.collections[0].get_paths()
        for i, p in enumerate(paths):
            v = p.vertices
            ra_points = v[:, 0]
            dec_points = v[:, 1]
            print(ra_points, ra_crit_list[i])
            npt.assert_almost_equal(ra_points[0], ra_crit_list[i][0], 5)
            npt.assert_almost_equal(dec_points[0], dec_crit_list[i][0], 5)

    def test_half_light_radius(self):
        kwargs_profile = [{'Rs': 0.16350224766074103, 'q': 0.4105628122365978, 'center_x': -0.019983826426838536,
            'center_y': 0.90000011282957304, 'phi_G': 0.14944144075912402, 'sigma0': 1.3168943578511678},
            {'Rs': 0.29187068596715743, 'q': 0.70799587973181288, 'center_x': 0.020568531548241405,
            'center_y': 0.036038490364800925, 'Ra': 0.020000382843298824, 'phi_G': -0.37221683730659516,
            'sigma0': 85.948773973262391}]
        kwargs_options = {'lens_model_list': ['SPEMD'], 'lens_model_internal_bool': [True], 'lens_light_model_internal_bool': [True, True], 'lens_light_model_list': ['HERNQUIST_ELLIPSE', 'PJAFFE_ELLIPSE']}
        lensAnalysis = LensAnalysis(kwargs_options, {})
        r_eff_true = 0.282786143932
        r_eff = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.05)
        #r_eff_new = lensAnalysis.half_light_radius(kwargs_profile, numPix=1000, deltaPix=0.01)
        npt.assert_almost_equal(r_eff/r_eff_true, 1, 2)

if __name__ == '__main__':
    pytest.main()
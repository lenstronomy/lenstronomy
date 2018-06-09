__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
import lenstronomy.Util.param_util as param_util


class TestLensModelExtensions(object):
    """
    tests the source model routines
    """
    def setup(self):
        pass

    def test_critical_curves(self):
        lens_model_list = ['SPEP']
        phi, q = 1., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        lensModel = LensModelExtensions(lens_model_list)
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModel.critical_curve_caustics(kwargs_lens,
                                                                                compute_window=5, grid_scale=0.005)
        print(ra_caustic_list)
        npt.assert_almost_equal(ra_caustic_list[0][3], -0.25629009803139047, decimal=5)
        npt.assert_almost_equal(dec_caustic_list[0][3], -0.39153358367275115, decimal=5)
        npt.assert_almost_equal(ra_crit_list[0][3], -0.53249999999999997, decimal=5)
        npt.assert_almost_equal(dec_crit_list[0][3], -1.2536936868024853, decimal=5)

    def test_critical_curves_tiling(self):
        lens_model_list = ['SPEP']
        phi, q = 1., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        lensModel = LensModelExtensions(lens_model_list)
        ra_crit, dec_crit = lensModel.critical_curve_tiling(kwargs_lens, compute_window=5, start_scale=0.01, max_order=10)
        npt.assert_almost_equal(ra_crit[0], -0.5355208333333333, decimal=5)

    def test_get_magnification_model(self):
        self.kwargs_options = { 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'],
                               'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]

        x_pos = np.array([1., 1., 2.])
        y_pos = np.array([-1., 0., 0.])
        lens_model = LensModelExtensions(lens_model_list=['GAUSSIAN'])
        mag = lens_model.magnification_finite(x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_elliptical_ray_trace(self):

        lens_model_list = ['SPEMD','SHEAR']

        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': 0.02, 'e2': -0.09, 'center_x': 0, 'center_y': 0},{'e1':0.01,'e2':0.03}]

        extension = LensModelExtensions(lens_model_list)
        x_image, y_image = [ 0.56153533,-0.78067875,-0.72551184,0.75664112],[-0.74722528,0.52491177,-0.72799235,0.78503659]

        mag_square_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                         grid_number=200,window_size=0.1)

        mag_polar_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                        grid_number=200,window_size=0.1,polar_grid=True)

        npt.assert_almost_equal(mag_polar_grid,mag_square_grid,decimal=5)

    def test_profile_slope(self):
        lens_model = LensModelExtensions(lens_model_list=['SPP'])
        gamma_in = 2.
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)
        gamma_in = 1.7
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        gamma_in = 2.5
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma_in, 'center_x': 0, 'center_y': 0}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        lens_model = LensModelExtensions(lens_model_list=['SPEP'])
        gamma_in = 2.
        phi, q = 0.34403343049704888, 0.89760957136967312
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1.4516812130749424, 'e1': e1, 'e2': e2, 'center_x': -0.04507598845306314,
         'center_y': 0.054491803177414651, 'gamma': gamma_in}]
        gamma_out = lens_model.profile_slope(kwargs_lens)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

    def test_lens_center(self):
        center_x, center_y = 0.43, -0.67
        kwargs_lens = [{'theta_E': 1, 'center_x': center_x, 'center_y': center_y}]
        lensModel = LensModelExtensions(lens_model_list=['SIS'])
        center_x_out, center_y_out = lensModel.lens_center(kwargs_lens)
        npt.assert_almost_equal(center_x_out, center_x, 2)
        npt.assert_almost_equal(center_y_out, center_y, 2)

    def test_external_shear(self):
        lens_model_list = ['SHEAR']
        kwargs_lens = [{'e1': 0.1, 'e2': 0.01}]
        lensModel = LensModelExtensions(lens_model_list)
        phi, gamma = lensModel.external_shear(kwargs_lens)
        npt.assert_almost_equal(phi, 0.049834326245581012, decimal=8)
        npt.assert_almost_equal(gamma, 0.10049875621120891, decimal=8)

    def test_external_lensing_effect(self):
        lens_model_list = ['SHEAR']
        kwargs_lens = [{'e1': 0.1, 'e2': 0.01}]
        lensModel = LensModelExtensions(lens_model_list)
        alpha0_x, alpha0_y, kappa_ext, shear1, shear2 = lensModel.external_lensing_effect(kwargs_lens, lens_model_internal_bool=[False])
        print(alpha0_x, alpha0_y, kappa_ext, shear1, shear2)
        assert alpha0_x == 0
        assert alpha0_y == 0
        assert shear1 == 0.1
        assert shear2 == 0.01
        assert kappa_ext == 0


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

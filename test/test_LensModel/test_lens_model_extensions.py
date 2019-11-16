__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
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
        lensModel = LensModelExtensions(LensModel(lens_model_list))
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
        lensModel = LensModelExtensions(LensModel(lens_model_list))
        ra_crit, dec_crit = lensModel.critical_curve_tiling(kwargs_lens, compute_window=5, start_scale=0.01, max_order=10)
        npt.assert_almost_equal(ra_crit[0], -0.5355208333333333, decimal=5)

    def test_get_magnification_model(self):
        self.kwargs_options = { 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'],
                               'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]

        x_pos = np.array([1., 1., 2.])
        y_pos = np.array([-1., 0., 0.])
        lens_model = LensModelExtensions(LensModel(lens_model_list=['GAUSSIAN']))
        mag = lens_model.magnification_finite(x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_elliptical_ray_trace(self):

        lens_model_list = ['SPEMD','SHEAR']

        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': 0.02, 'e2': -0.09, 'center_x': 0, 'center_y': 0},
                       {'gamma1':0.01, 'gamma2':0.03}]

        extension = LensModelExtensions(LensModel(lens_model_list))
        x_image, y_image = [ 0.56153533,-0.78067875,-0.72551184,0.75664112],[-0.74722528,0.52491177,-0.72799235,0.78503659]

        mag_square_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                         grid_number=200,window_size=0.1)

        mag_polar_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                        grid_number=200,window_size=0.1,polar_grid=True)

        npt.assert_almost_equal(mag_polar_grid,mag_square_grid,decimal=5)

    def test_zoom_source(self):
        lens_model_list = ['SPEMD', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        x_source, y_source = 0.02, 0.01
        kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': 0.1, 'gamma': 2, 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.05, 'gamma2': -0.03}]

        x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source,
                                                                     sourcePos_y=y_source)

        image = lensModelExtensions.zoom_source(x_img[0], y_img[0], kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                    shape="GAUSSIAN")
        assert len(image) == 100


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

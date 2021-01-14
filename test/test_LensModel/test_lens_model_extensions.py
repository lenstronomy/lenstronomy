__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Cosmo.background import Background
import lenstronomy.Util.param_util as param_util
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM

class TestLensModelExtensions(object):
    """
    tests the source model routines
    """
    def setup(self):

        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    def test_critical_curves(self):
        lens_model_list = ['SPEP']
        phi, q = 1., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        lens_model = LensModel(lens_model_list)
        lensModelExtensions = LensModelExtensions(LensModel(lens_model_list))
        ra_crit_list, dec_crit_list, ra_caustic_list, dec_caustic_list = lensModelExtensions.critical_curve_caustics(kwargs_lens,
                                                                                                           compute_window=5, grid_scale=0.005)

        # here we test whether the caustic points are in fact at high magnifications (close to infinite)
        # close here means above magnification of 1000000 (with matplotlib method, this limit achieved was 170)
        for k in range(len(ra_crit_list)):
            ra_crit = ra_crit_list[k]
            dec_crit = dec_crit_list[k]
            mag = lens_model.magnification(ra_crit, dec_crit, kwargs_lens)
            assert np.all(np.abs(mag) > 100000)

    def test_critical_curves_tiling(self):
        lens_model_list = ['SPEP']
        phi, q = 1., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        lensModel = LensModelExtensions(LensModel(lens_model_list))
        ra_crit, dec_crit = lensModel.critical_curve_tiling(kwargs_lens, compute_window=5, start_scale=0.01,
                                                            max_order=10)
        # here we test whether the caustic points are in fact at high magnifications (close to infinite)
        # close here means above magnification of 1000. This is more precise than the critical_curve_caustics() method
        lens_model = LensModel(lens_model_list)
        mag = lens_model.magnification(ra_crit, dec_crit, kwargs_lens)
        assert np.all(np.abs(mag) > 1000)

    def test_get_magnification_model(self):
        self.kwargs_options = { 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'],
                               'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]

        x_pos = np.array([1., 1., 2.])
        y_pos = np.array([-1., 0., 0.])
        lens_model = LensModelExtensions(LensModel(lens_model_list=['GAUSSIAN']))
        mag = lens_model.magnification_finite(x_pos, y_pos, kwargs_lens, source_sigma=0.003, window_size=0.1,
                                              grid_number=100)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_elliptical_ray_trace(self):

        lens_model_list = ['SPEP','SHEAR']

        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': 0.02, 'e2': -0.09, 'center_x': 0, 'center_y': 0},
                       {'gamma1':0.01, 'gamma2':0.03}]

        extension = LensModelExtensions(LensModel(lens_model_list))
        x_image = [0.56153533, -0.78067875, -0.72551184, 0.75664112]
        y_image = [-0.74722528, 0.52491177, -0.72799235, 0.78503659]

        mag_square_grid = extension.magnification_finite(x_image, y_image, kwargs_lens, source_sigma=0.001,
                                                         grid_number=200, window_size=0.1)
        mag_polar_grid = extension.magnification_finite(x_image, y_image, kwargs_lens, source_sigma=0.001,
                                                        grid_number=200, window_size=0.1, polar_grid=True)

        npt.assert_almost_equal(mag_polar_grid,mag_square_grid,decimal=5)

    def test_magnification_finite_adaptive(self):

        lens_model_list = ['EPL', 'SHEAR']
        z_source = 1.5
        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': 0.02, 'e2': -0.09, 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.01, 'gamma2': 0.03}]

        lensmodel = LensModel(lens_model_list)
        extension = LensModelExtensions(lensmodel)
        solver = LensEquationSolver(lensmodel)
        source_x, source_y = 0.07, 0.03
        x_image, y_image = solver.findBrightImage(source_x, source_y, kwargs_lens)

        source_fwhm_parsec = 40.

        pc_per_arcsec = 1000 / self.cosmo.arcsec_per_kpc_proper(z_source).value
        source_sigma = source_fwhm_parsec / pc_per_arcsec / 2.355

        mag_square_grid = extension.magnification_finite(x_image, y_image, kwargs_lens, source_sigma=source_sigma,
                                                          grid_number=1501, window_size=0.15)

        mag_adaptive_grid = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens, source_fwhm_parsec,
                                                                    z_source, cosmo=self.cosmo, tol=0.0001)

        mag_point_source = abs(lensmodel.magnification(x_image, y_image, kwargs_lens))

        npt.assert_almost_equal(mag_square_grid/mag_adaptive_grid, np.ones_like(mag_square_grid), 2)
        npt.assert_almost_equal(mag_adaptive_grid/mag_point_source, np.ones_like(mag_square_grid), 2)

        flux_array = np.array([0., 0.])
        x_image, y_image = [x_image[0]], [y_image[0]]
        grid_x = np.array([0., source_sigma])
        grid_y = np.array([0., 0.])
        grid_r = np.hypot(grid_x, grid_y)

        source_model = LightModel(['GAUSSIAN'])
        kwargs_source = [{'amp': 1., 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma}]

        r_min = 0.
        r_max = source_sigma*0.9
        flux_array = extension._magnification_adaptive_iteration(flux_array, x_image, y_image, grid_x, grid_y, grid_r, r_min, r_max,
                                                                 lensmodel, kwargs_lens, source_model, kwargs_source)
        bx, by = lensmodel.ray_shooting(x_image[0], y_image[0], kwargs_lens)
        sb_true = source_model.surface_brightness(bx, by, kwargs_source)
        npt.assert_equal(True, flux_array[0] == sb_true)
        npt.assert_equal(True, flux_array[1] == 0.)

        r_min = source_sigma*0.9
        r_max = 2 * source_sigma

        flux_array = extension._magnification_adaptive_iteration(flux_array, x_image, y_image, grid_x, grid_y, grid_r, r_min, r_max,
                                                                 lensmodel, kwargs_lens, source_model, kwargs_source)
        bx, by = lensmodel.ray_shooting(x_image[0] + source_sigma, y_image[0], kwargs_lens)
        sb_true = source_model.surface_brightness(bx, by, kwargs_source)
        npt.assert_equal(True, flux_array[1] == sb_true)

    def test_zoom_source(self):
        lens_model_list = ['SIE', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        x_source, y_source = 0.02, 0.01
        kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': 0.1, 'center_x': 0, 'center_y': 0},
                       {'gamma1': 0.05, 'gamma2': -0.03}]

        x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source,
                                                                     sourcePos_y=y_source)

        image = lensModelExtensions.zoom_source(x_img[0], y_img[0], kwargs_lens, source_sigma=0.003, window_size=0.1,
                                                grid_number=100, shape="GAUSSIAN")
        assert len(image) == 100


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

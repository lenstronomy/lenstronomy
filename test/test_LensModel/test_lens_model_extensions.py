__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Util.magnification_finite_util import auto_raytracing_grid_resolution, auto_raytracing_grid_size
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

    def test_magnification_finite(self):

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

        grid_size = auto_raytracing_grid_size(source_fwhm_parsec)
        grid_resolution = auto_raytracing_grid_resolution(source_fwhm_parsec)
        # make this even higher resolution to show convergence
        grid_number = int(2 * grid_size / grid_resolution)
        window_size = 2 * grid_size

        mag_square_grid = extension.magnification_finite(x_image, y_image, kwargs_lens, source_sigma=source_sigma,
                                                         grid_number=grid_number, window_size=window_size)
        flux_ratios_square_grid = mag_square_grid / max(mag_square_grid)

        mag_adaptive_grid = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                                    source_fwhm_parsec,
                                                                    z_source, cosmo=self.cosmo)
        flux_ratios_adaptive_grid = mag_adaptive_grid / max(mag_adaptive_grid)

        mag_adaptive_grid_fixed_aperture_size = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                    source_fwhm_parsec,
                                                    z_source, cosmo=self.cosmo, fixed_aperture_size=True,
                                                    grid_radius_arcsec=0.2)
        flux_ratios_fixed_aperture_size = mag_adaptive_grid_fixed_aperture_size / max(mag_adaptive_grid_fixed_aperture_size)

        mag_adaptive_grid_2 = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                                      source_fwhm_parsec, z_source,
                                                                      cosmo=self.cosmo, axis_ratio=0)
        mag_adaptive_grid_3 = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                                      source_fwhm_parsec, z_source,
                                                                      cosmo=self.cosmo, axis_ratio=0)

        flux_ratios_adaptive_grid_2 = mag_adaptive_grid_2 / max(mag_adaptive_grid_2)
        flux_ratios_adaptive_grid_3 = mag_adaptive_grid_3 / max(mag_adaptive_grid_3)

        # tests the default cosmology
        _ = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                    source_fwhm_parsec,
                                                    z_source, cosmo=None, tol=0.0001)

        # test smallest eigenvalue
        _ = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                    source_fwhm_parsec,
                                                    z_source, cosmo=None, tol=0.0001, use_largest_eigenvalue=False)

        # tests the r_max > sqrt(2) * grid_radius stop criterion
        _ = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                    source_fwhm_parsec,
                                                    z_source, cosmo=None, tol=0.0001, step_size=1000)

        mag_point_source = abs(lensmodel.magnification(x_image, y_image, kwargs_lens))

        quarter_precent_precision = [0.0025] * 4
        npt.assert_array_less(flux_ratios_square_grid / flux_ratios_adaptive_grid - 1,
                              quarter_precent_precision)
        npt.assert_array_less(flux_ratios_fixed_aperture_size / flux_ratios_adaptive_grid - 1,
                              quarter_precent_precision)
        npt.assert_array_less(flux_ratios_square_grid / flux_ratios_adaptive_grid_2 - 1,
                              quarter_precent_precision)
        npt.assert_array_less(flux_ratios_adaptive_grid_3 / flux_ratios_adaptive_grid_2 - 1,
                              quarter_precent_precision)
        half_percent_precision = [0.005] * 4
        npt.assert_array_less(mag_square_grid / mag_adaptive_grid - 1, half_percent_precision)
        npt.assert_array_less(mag_square_grid / mag_adaptive_grid_2 - 1, half_percent_precision)
        npt.assert_array_less(mag_adaptive_grid / mag_point_source - 1, half_percent_precision)

        flux_array = np.array([0., 0.])
        grid_x = np.array([0., source_sigma])
        grid_y = np.array([0., 0.])
        grid_r = np.hypot(grid_x, grid_y)

        # test that the double gaussian profile has 2x the flux when dx, dy = 0
        magnification_double_gaussian = extension.magnification_finite_adaptive(x_image, y_image, source_x, source_y, kwargs_lens,
                                                                    source_fwhm_parsec, z_source, cosmo=self.cosmo,
                                                      source_light_model='DOUBLE_GAUSSIAN', dx=0., dy=0., amp_scale=1., size_scale=1.)
        npt.assert_almost_equal(magnification_double_gaussian, 2 * mag_adaptive_grid)


        grid_radius = 0.3
        npix = 400
        _x = _y = np.linspace(-grid_radius, grid_radius, npix)
        resolution = 2 * grid_radius / npix
        xx, yy = np.meshgrid(_x, _y)
        for i in range(0, 4):
            beta_x, beta_y = lensmodel.ray_shooting(x_image[i] + xx.ravel(), y_image[i] + yy.ravel(), kwargs_lens)
            source_light_model = LightModel(['GAUSSIAN'] * 2)
            amp_scale, dx, dy, size_scale = 0.2, 0.005, -0.005, 1.
            kwargs_source = [{'amp': 1., 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma},
                             {'amp': amp_scale, 'center_x': source_x + dx, 'center_y': source_y + dy,
                              'sigma': source_sigma * size_scale}]
            sb = source_light_model.surface_brightness(beta_x, beta_y, kwargs_source)
            magnification_double_gaussian_reference = np.sum(sb) * resolution ** 2
            magnification_double_gaussian = extension.magnification_finite_adaptive([x_image[i]], [y_image[i]], source_x, source_y,
                                                                                    kwargs_lens,
                                                                                    source_fwhm_parsec, z_source,
                                                                                    cosmo=self.cosmo,
                                                                                    source_light_model='DOUBLE_GAUSSIAN',
                                                                                    dx=dx, dy=dy, amp_scale=amp_scale,
                                                                                    size_scale=size_scale,
                                                                                    grid_resolution=resolution,
                                                                                    grid_radius_arcsec=grid_radius,
                                                                                    axis_ratio=1.)
            npt.assert_almost_equal(magnification_double_gaussian/magnification_double_gaussian_reference, 1., 3)

        source_model = LightModel(['GAUSSIAN'])
        kwargs_source = [{'amp': 1., 'center_x': source_x, 'center_y': source_y, 'sigma': source_sigma}]

        r_min = 0.
        r_max = source_sigma * 0.9
        flux_array = extension._magnification_adaptive_iteration(flux_array, x_image[0], y_image[0], grid_x, grid_y, grid_r,
                                                                 r_min, r_max,
                                                                 lensmodel, kwargs_lens, source_model, kwargs_source)
        bx, by = lensmodel.ray_shooting(x_image[0], y_image[0], kwargs_lens)
        sb_true = source_model.surface_brightness(bx, by, kwargs_source)
        npt.assert_equal(True, flux_array[0] == sb_true)
        npt.assert_equal(True, flux_array[1] == 0.)

        r_min = source_sigma * 0.9
        r_max = 2 * source_sigma

        flux_array = extension._magnification_adaptive_iteration(flux_array, x_image[0], y_image[0], grid_x, grid_y, grid_r,
                                                                 r_min, r_max,
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

    def test_tangential_average(self):
        lens_model_list = ['SIS']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        tang_stretch_ave = lensModelExtensions.tangential_average(x=1.1, y=0, kwargs_lens=[{'theta_E': 1, 'center_x': 0, 'center_y': 0}], dr=1, smoothing=None, num_average=9)
        npt.assert_almost_equal(tang_stretch_ave, -2.525501464097973, decimal=6)

    def test_caustic_area(self):
        lens_model_list = ['SIE']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        kwargs_lens = [{'theta_E': 1, 'e1': 0.2, 'e2': 0, 'center_x': 0, 'center_y': 0}]
        kwargs_caustic_num = {'compute_window': 3, 'grid_scale': 0.005, 'center_x': 0, 'center_y': 0}

        area = lensModelExtensions.caustic_area(kwargs_lens=kwargs_lens, kwargs_caustic_num=kwargs_caustic_num,
                                                index_vertices=0)
        npt.assert_almost_equal(area, 0.08445866728739478, decimal=3)


if __name__ == '__main__':
    pytest.main("-k TestLensModel")

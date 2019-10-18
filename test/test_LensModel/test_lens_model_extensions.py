__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.curved_arc import CurvedArc
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util import util


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

        kwargs_lens = [{'theta_E': 1., 'gamma': 2., 'e1': 0.02, 'e2': -0.09, 'center_x': 0, 'center_y': 0},{'e1':0.01,'e2':0.03}]

        extension = LensModelExtensions(LensModel(lens_model_list))
        x_image, y_image = [ 0.56153533,-0.78067875,-0.72551184,0.75664112],[-0.74722528,0.52491177,-0.72799235,0.78503659]

        mag_square_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                         grid_number=200,window_size=0.1)

        mag_polar_grid = extension.magnification_finite(x_image,y_image,kwargs_lens,source_sigma=0.001,
                                                        grid_number=200,window_size=0.1,polar_grid=True)

        npt.assert_almost_equal(mag_polar_grid,mag_square_grid,decimal=5)

    def test_profile_slope(self):
        lens_model = LensModelExtensions(LensModel(lens_model_list=['SPP']))
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

        kwargs_lens_bad = [{'theta_E': 100, 'gamma': 2, 'center_x': 0, 'center_y': 0}]
        gamma_out_bad = lens_model.profile_slope(kwargs_lens_bad, verbose=False)
        assert np.isnan(gamma_out_bad)

        lens_model = LensModelExtensions(LensModel(lens_model_list=['SPEP']))
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
        lensModel = LensModelExtensions(LensModel(lens_model_list=['SIS']))
        center_x_out, center_y_out = lensModel.lens_center(kwargs_lens)
        npt.assert_almost_equal(center_x_out, center_x, 2)
        npt.assert_almost_equal(center_y_out, center_y, 2)

    def test_effective_einstein_radius(self):
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        lensModel = LensModelExtensions(LensModel(lens_model_list=['SIS']))
        ret = lensModel.effective_einstein_radius(kwargs_lens,
                                                  get_precision=True)

        assert len(ret) == 2
        npt.assert_almost_equal(ret[0], 1., decimal=2)
        kwargs_lens_bad = [{'theta_E': 100, 'center_x': 0, 'center_y': 0}]
        ret_nan = lensModel.effective_einstein_radius(kwargs_lens_bad,
                                                      get_precision=True, verbose=False)
        assert np.isnan(ret_nan)

    def test_external_shear(self):
        lens_model_list = ['SHEAR']
        kwargs_lens = [{'e1': 0.1, 'e2': 0.01}]
        lensModel = LensModelExtensions(LensModel(lens_model_list))
        phi, gamma = lensModel.external_shear(kwargs_lens)
        npt.assert_almost_equal(phi, 0.049834326245581012, decimal=8)
        npt.assert_almost_equal(gamma, 0.10049875621120891, decimal=8)

    def test_external_lensing_effect(self):
        lens_model_list = ['SHEAR']
        kwargs_lens = [{'e1': 0.1, 'e2': 0.01}]
        lensModel = LensModelExtensions(LensModel(lens_model_list))
        alpha0_x, alpha0_y, kappa_ext, shear1, shear2 = lensModel.external_lensing_effect(kwargs_lens, lens_model_internal_bool=[False])
        print(alpha0_x, alpha0_y, kappa_ext, shear1, shear2)
        assert alpha0_x == 0
        assert alpha0_y == 0
        assert shear1 == 0.1
        assert shear2 == 0.01
        assert kappa_ext == 0

    def test_zoom_source(self):
        lens_model_list = ['SPEMD', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        x_source, y_source = 0.02, 0.01
        kwargs_lens = [{'theta_E': 1, 'e1': 0.1, 'e2': 0.1, 'gamma': 2, 'center_x': 0, 'center_y': 0},
                       {'e1': 0.05, 'e2': -0.03}]

        x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source,
                                                                     sourcePos_y=y_source)

        image = lensModelExtensions.zoom_source(x_img[0], y_img[0], kwargs_lens, source_sigma=0.003, window_size=0.1, grid_number=100,
                    shape="GAUSSIAN")
        assert len(image) == 100

    def test_radial_tangential_distortions(self):
        lens_model_list = ['CURVED_ARC', 'SHEAR', 'FLEXION']
        center_x, center_y = 0, 0
        curvature = 1./2
        lens = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'tangential_stretch': 10, 'radial_stretch': 1., 'curvature': curvature,
                        'direction': -10, 'center_x': center_x, 'center_y': center_y},
                       {'e1': -0., 'e2': -0.0},
                       {'g1': 0., 'g2': 0., 'g3': -0., 'g4': 0}]

        extensions = LensModelExtensions(lensModel=lens)

        radial_stretch, tangential_stretch, d_tang_d_tang, d_tang_d_rad, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(
            x=center_x, y=center_y, kwargs_lens=kwargs_lens, smoothing_3rd=0.0001)

        l = 1. / d_angle_d_tang
        npt.assert_almost_equal(l, 1./curvature)

    def test_radial_tangential_differentials(self):
        lens_model_list = ['CURVED_ARC', 'SHEAR', 'FLEXION']
        center_x, center_y = 0, 0
        curvature = 1./2
        lens = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'tangential_stretch': 10, 'radial_stretch': 1., 'curvature': curvature,
                        'direction': -10, 'center_x': center_x, 'center_y': center_y},
                       {'e1': -0., 'e2': -0.0},
                       {'g1': 0., 'g2': 0., 'g3': -0., 'g4': 0}]

        extensions = LensModelExtensions(lensModel=lens)
        from lenstronomy.Util import util
        x, y = util.make_grid(numPix=10, deltapix=1)
        radial_stretch, tangential_stretch, d_tang_d_tang, d_tang_d_rad, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(x, y, kwargs_lens, smoothing_3rd=0.001)
        npt.assert_almost_equal(np.sum(d_angle_d_rad), 0, decimal=3)
        npt.assert_almost_equal(np.sum(d_rad_d_rad), 0, decimal=3)

        lens_model_list = ['SIS']
        center_x, center_y = 0, 0
        lens = LensModel(lens_model_list=lens_model_list)
        kwargs_lens = [{'theta_E': 1, 'center_x': center_x, 'center_y': center_y}]

        extensions = LensModelExtensions(lensModel=lens)
        radial_stretch, tangential_stretch, d_tang_d_tang, d_tang_d_rad, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(2, 2, kwargs_lens, smoothing_3rd=0.001)

        npt.assert_almost_equal(radial_stretch, 1, decimal=5)
        print(d_tang_d_tang, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad)
        npt.assert_almost_equal(tangential_stretch, 1.5469181606780271, decimal=5)

        radial_stretch, tangential_stretch, d_tang_d_tang, d_tang_d_rad, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(
            np.array([2]), np.array([2]), kwargs_lens, smoothing_3rd=0.001)

        npt.assert_almost_equal(radial_stretch, 1, decimal=5)
        print(d_tang_d_tang, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad)
        npt.assert_almost_equal(tangential_stretch, 1.5469181606780271, decimal=5)

        mag = lens.magnification(x, y, kwargs_lens)
        radial_stretch, tangential_stretch, d_tang_d_tang, d_tang_d_rad, d_angle_d_tang, d_rad_d_rad, d_angle_d_rad, orientation_angle = extensions.radial_tangential_differentials(
            x, y, kwargs_lens, smoothing_3rd=0.001)
        mag_tang_rad = tangential_stretch * radial_stretch
        npt.assert_almost_equal(mag_tang_rad, mag, decimal=5)

    def test_curved_arc_estimate(self):
        lens_model_list = ['SPP']
        lens = LensModel(lens_model_list=lens_model_list)
        arc = LensModel(lens_model_list=['CURVED_ARC'])
        theta_E = 4
        gamma = 2.
        kwargs_lens = [{'theta_E': theta_E, 'gamma': gamma, 'center_x': 0, 'center_y': 0}]
        ext = LensModelExtensions(lensModel=lens)
        x_0, y_0 = 5, 0
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc = CurvedArc.stretch2spp(**kwargs_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)
        x, y = util.make_grid(numPix=10, deltapix=1)
        alpha_x, alpha_y = lens.alpha(x, y, kwargs_lens)
        alpha0_x, alpha0_y = lens.alpha(x_0, y_0, kwargs_lens)
        alpha_x_arc, alpha_y_arc = arc.alpha(x, y, [kwargs_arc])
        npt.assert_almost_equal(alpha_x_arc, alpha_x - alpha0_x, decimal=3)
        npt.assert_almost_equal(alpha_y_arc, alpha_y - alpha0_y, decimal=3)

        x_0, y_0 = 0., 3
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc = CurvedArc.stretch2spp(**kwargs_arc)
        print(kwargs_arc)
        print(theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)

        x_0, y_0 = -2, -3
        kwargs_arc = ext.curved_arc_estimate(x_0, y_0, kwargs_lens)
        theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc = CurvedArc.stretch2spp(**kwargs_arc)
        npt.assert_almost_equal(theta_E_arc, theta_E, decimal=4)
        npt.assert_almost_equal(gamma_arc, gamma, decimal=3)
        npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
        npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)

    def test_arcs_at_image_position(self):
        # lensing quantities
        kwargs_shear = {'e1': 0.02, 'e2': -0.04}  # shear values to the source plane
        kwargs_spp = {'theta_E': 1.26, 'gamma': 2., 'e1': 0.1, 'e2': -0.1, 'center_x': 0.0, 'center_y': 0.0}  # parameters of the deflector lens model

        # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list = ['SPEP']  #, 'SHEAR']
        kwargs_lens = [kwargs_spp]  #, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_class)
        source_x = 0.
        source_y = 0.05
        x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens, numImages=4,
                                                              min_distance=0.05, search_window=5)
        arc_model = LensModel(lens_model_list=['CURVED_ARC', 'SHIFT'])
        for i in range(len(x_image)):
            x0, y0 = x_image[i], y_image[i]
            print(x0, y0, i)
            ext = LensModelExtensions(lensModel=lens_model_class)
            kwargs_arc_i = ext.curved_arc_estimate(x0, y0, kwargs_lens)
            alpha_x, alpha_y = lens_model_class.alpha(x0, y0, kwargs_lens)
            kwargs_arc = [kwargs_arc_i, {'alpha_x': alpha_x, 'alpha_y': alpha_y}]
            print(kwargs_arc_i)
            direction = kwargs_arc_i['direction']
            print(np.cos(direction), np.sin(direction))
            x, y = util.make_grid(numPix=5, deltapix=0.01)
            x = x0
            y = y0
            gamma1_arc, gamma2_arc = arc_model.gamma(x, y, kwargs_arc)
            gamma1, gamma2 = lens_model_class.gamma(x, y, kwargs_lens)
            print(gamma1, gamma2)
            npt.assert_almost_equal(gamma1_arc, gamma1, decimal=3)
            npt.assert_almost_equal(gamma2_arc, gamma2, decimal=3)
            theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc = CurvedArc.stretch2spp(**kwargs_arc_i)
            print(theta_E_arc, gamma_arc, center_x_spp_arc, center_y_spp_arc)
            npt.assert_almost_equal(center_x_spp_arc, 0, decimal=3)
            npt.assert_almost_equal(center_y_spp_arc, 0, decimal=3)



if __name__ == '__main__':
    pytest.main("-k TestLensModel")

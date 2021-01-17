import numpy as np
import numpy.testing as npt
from lenstronomy.LensModel.QuadOptimizer.multi_scale_model import MultiScaleModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

class TestMultiScaleModel(object):

    def setup(self):

        zlens, zsource = 0.45, 1.5
        self.zlens, self.zsource = zlens, zsource
        self.lens_model_list_other = ['NFW_MC'] * 12
        self.redshift_list_other = [0.2, 0.2, 0.4, 0.4, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 1., 1.]
        redshift_list = [zlens] * 2 + self.redshift_list_other
        lens_model_list = ['EPL', 'SHEAR'] + self.lens_model_list_other
        c = 16.
        self.kwargs_lens_other = [{'logM': 9, 'concentration': c, 'center_x': 0.5, 'center_y': -0.09},
                        {'logM': 9., 'concentration': c, 'center_x': -0.9, 'center_y': 0.99},
                        {'logM': 9., 'concentration': c, 'center_x': 1.5, 'center_y': -1.09},
                        {'logM': 9., 'concentration': c, 'center_x': 0.59, 'center_y': 0.85}, # close to image 2
                        {'logM': 9., 'concentration': c, 'center_x': 0.21, 'center_y': -1.03}, # close to image 1
                        {'logM': 9., 'concentration': c, 'center_x': 1.04, 'center_y': 0.69},
                        {'logM': 9., 'concentration': c, 'center_x': 0.29, 'center_y': -0.3},
                        {'logM': 9., 'concentration': c, 'center_x': -0.87, 'center_y': 0.8},
                        {'logM': 9., 'concentration': c, 'center_x': -0.1, 'center_y': -0.7},
                        {'logM': 9., 'concentration': c, 'center_x': 0.35, 'center_y': -1.2},
                        {'logM': 9., 'concentration': c, 'center_x': -0.95, 'center_y': 1.},
                        {'logM': 9., 'concentration': c, 'center_x': 1.25, 'center_y': 0.45}]

        self.kwargs_lens = [{'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.2, 'e2': 0., 'gamma': 2.},
                       {'gamma1': 0.05, 'gamma2': 0.02}] + self.kwargs_lens_other

        self.lens_model = LensModel(lens_model_list, lens_redshift_list=redshift_list,
                                    multi_plane=True, z_source=zsource)
        self.extension = LensModelExtensions(self.lens_model)

        solver = LensEquationSolver(self.lens_model)
        self.source_x, self.source_y = 0.06, -0.03
        self.x_image, self.y_image = solver.findBrightImage(self.source_x, self.source_y, self.kwargs_lens)

    def test_kwargs_shift(self):

        for i in range(0, 4):
            model = MultiScaleModel(self.x_image[i], self.y_image[i], self.source_x, self.source_y,
                                      self.lens_model_list_other, self.redshift_list_other, self.kwargs_lens_other,
                                      self.zlens, self.zsource)
            kwargs_shift = model.compute_kwargs_shift()

            lens_model_list_shift = ['SHIFT'] + self.lens_model_list_other
            redshift_list_shift = [self.zlens] + self.redshift_list_other
            kwargs_list_shift = [kwargs_shift] + self.kwargs_lens_other
            lens_model_shift = LensModel(lens_model_list_shift, lens_redshift_list=redshift_list_shift,
                                         z_source=self.zsource, multi_plane=True)
            beta_x, beta_y = lens_model_shift.ray_shooting(self.x_image[i], self.y_image[i], kwargs_list_shift)
            npt.assert_almost_equal(beta_x, self.source_x)
            npt.assert_almost_equal(beta_y, self.source_y)

    def test_hessian_fit(self):

        for i in range(0, 4):

            fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x_image[i], self.y_image[i],
                                                                 self.kwargs_lens)

            hessian_init = {'f_xx': fxx, 'f_xy': fxy, 'f_yx': fyx, 'f_yy': fyy, 'ra_0': self.x_image[i],
                            'dec_0': self.y_image[i]}

            model = MultiScaleModel(self.x_image[i], self.y_image[i], self.source_x, self.source_y,
                                    self.lens_model_list_other, self.redshift_list_other, self.kwargs_lens_other,
                                    self.zlens, self.zsource)

            angular_matching_scale = 1e-4
            fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x_image[i],
                                                         self.y_image[i],
                                                         self.kwargs_lens,
                                                         diff=angular_matching_scale)
            kappa_constraint = 0.5 * (fxx + fyy)
            gamma_constraint1 = 0.5 * (fxx - fyy)
            gamma_constraint2 = 0.5 * (fxy + fyx)

            kwargs_hessian, kwargs_fit, result = model.solve_kwargs_hessian(hessian_init, kappa_constraint, gamma_constraint1, gamma_constraint2,
                                   angular_matching_scale)

            lens_model_list_hessian = ['HESSIAN', 'SHIFT'] + self.lens_model_list_other
            redshift_list_arc = [self.zlens, self.zlens] + self.redshift_list_other
            lens_model_arc = LensModel(lens_model_list_hessian, lens_redshift_list=redshift_list_arc, z_source=self.zsource,
                                       multi_plane=True)
            beta_x, beta_y = lens_model_arc.ray_shooting(self.x_image[i], self.y_image[i], kwargs_fit)
            npt.assert_almost_equal(beta_x, self.source_x)
            npt.assert_almost_equal(beta_y, self.source_y)

            fxx_fit, fxy_fit, fyx_fit, fyy_fit = lens_model_arc.hessian(self.x_image[i],
                                                                        self.y_image[i],
                                                                        kwargs_fit,
                                                                        diff=angular_matching_scale)
            kappa_fit = 0.5 * (fxx_fit + fyy_fit)
            gamma1_fit = 0.5 * (fxx_fit - fyy_fit)
            gamma2_fit = 0.5 * (fxy_fit + fyx_fit)
            npt.assert_almost_equal(kappa_constraint, kappa_fit, 2)
            npt.assert_almost_equal(gamma_constraint1, gamma1_fit, 2)
            npt.assert_almost_equal(gamma_constraint2, gamma2_fit, 2)

    def test_curved_arc_fit(self):

        for fit_setting in ['FIXED_CURVATURE_DIRECTION', 'FIXED_CURVATURE']:
            for i in range(0, 4):

                curved_arc_init = self.extension.curved_arc_estimate(self.x_image[i], self.y_image[i],
                                                                     self.kwargs_lens)
                model = MultiScaleModel(self.x_image[i], self.y_image[i], self.source_x, self.source_y,
                                        self.lens_model_list_other, self.redshift_list_other, self.kwargs_lens_other,
                                        self.zlens, self.zsource)

                angular_matching_scale = 1e-4
                fxx, fxy, fyx, fyy = self.lens_model.hessian(self.x_image[i],
                                                             self.y_image[i],
                                                             self.kwargs_lens,
                                                             diff=angular_matching_scale)
                kappa_constraint = 0.5 * (fxx + fyy)
                gamma_constraint1 = 0.5 * (fxx - fyy)
                gamma_constraint2 = 0.5 * (fxy + fyx)

                kwargs_curved_arc, kwargs_fit, result = model.solve_kwargs_arc(curved_arc_init, kappa_constraint, gamma_constraint1, gamma_constraint2,
                                       angular_matching_scale, fit_setting)

                if fit_setting == 'FIXED_CURVATURE_DIRECTION':
                    npt.assert_equal(True, kwargs_curved_arc['direction'] == curved_arc_init['direction'])
                    npt.assert_equal(True, kwargs_curved_arc['curvature'] == curved_arc_init['curvature'])
                elif fit_setting == 'FIXED_DIRECTION':
                    npt.assert_equal(True, kwargs_curved_arc['direction'] == curved_arc_init['direction'])

                lens_model_list_arc = ['CURVED_ARC', 'SHIFT'] + self.lens_model_list_other
                redshift_list_arc = [self.zlens, self.zlens] + self.redshift_list_other
                lens_model_arc = LensModel(lens_model_list_arc, lens_redshift_list=redshift_list_arc, z_source=self.zsource,
                                           multi_plane=True)
                beta_x, beta_y = lens_model_arc.ray_shooting(self.x_image[i], self.y_image[i], kwargs_fit)
                npt.assert_almost_equal(beta_x, self.source_x)
                npt.assert_almost_equal(beta_y, self.source_y)

                fxx_fit, fxy_fit, fyx_fit, fyy_fit = lens_model_arc.hessian(self.x_image[i],
                                                                            self.y_image[i],
                                                                            kwargs_fit,
                                                                            diff=angular_matching_scale)
                kappa_fit = 0.5 * (fxx_fit + fyy_fit)
                gamma1_fit = 0.5 * (fxx_fit - fyy_fit)
                gamma2_fit = 0.5 * (fxy_fit + fyx_fit)
                npt.assert_almost_equal(kappa_constraint, kappa_fit, 2)
                npt.assert_almost_equal(gamma_constraint1, gamma1_fit, 2)
                npt.assert_almost_equal(gamma_constraint2, gamma2_fit, 2)


t = TestMultiScaleModel()
t.setup()
t.test_hessian_fit()

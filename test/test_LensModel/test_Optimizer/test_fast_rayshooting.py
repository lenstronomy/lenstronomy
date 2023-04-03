import numpy as np
from lenstronomy.LensModel.QuadOptimizer.multi_plane_fast import MultiplaneFast
from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFreeShear
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import numpy.testing as npt
import pytest


class TestFastRayShooting(object):

    def setup_method(self):

        self.zlens, self.zsource = 0.5, 1.5
        epl_kwargs = {'theta_E': 0.8, 'center_x': 0.1, 'center_y': 0., 'e1': -0.2, 'e2': 0.1, 'gamma': 2.05}
        shear_kwargs = {'gamma1': 0.09, 'gamma2': -0.02}
        kwargs_macro = [epl_kwargs, shear_kwargs]

        self.x_image = np.array([0.65043538, -0.31109505, 0.78906059, -0.86222271])
        self.y_image = np.array([-0.89067493, 0.94851787, 0.52882605, -0.25403778])

        halo_list = ['SIS', 'SIS', 'SIS']
        halo_z = [self.zlens - 0.1, self.zlens, self.zlens + 0.4]
        halo_kwargs = [{'theta_E': 0.05, 'center_x': 0.3, 'center_y': -0.9},
                       {'theta_E': 0.01, 'center_x': 1.3, 'center_y': -0.5},
                       {'theta_E': 0.02, 'center_x': -0.4, 'center_y': -0.4}]

        self.kwargs_epl = kwargs_macro + halo_kwargs
        self.zlist_epl = [self.zlens, self.zlens] + halo_z
        self.lens_model_list_epl = ['EPL', 'SHEAR'] + halo_list

        self.lensModel = LensModel(self.lens_model_list_epl, self.zlens, self.zsource, self.zlist_epl,
                              multi_plane=True)

        self.param_class = PowerLawFreeShear(self.kwargs_epl)

    def test_rayshooting(self):

        solver = LensEquationSolver(self.lensModel)
        source_x, source_y = -0.05, -0.02
        x_image_true, y_image_true = solver.findBrightImage(source_x, source_y, self.kwargs_epl)

        fast_rayshooting = MultiplaneFast(x_image_true, y_image_true, self.zlens, self.zsource,
                                           self.lens_model_list_epl, self.zlist_epl,
                 astropy_instance=None, param_class=self.param_class, foreground_rays=None,
                 tol_source=1e-5, numerical_alpha_class=None)

        x_fore, y_fore, alpha_x_fore, alpha_y_fore = fast_rayshooting._ray_shooting_fast_foreground()
        xtrue, ytrue, alpha_xtrue, alpha_ytrue = self.lensModel.lens_model.\
            ray_shooting_partial_comoving(np.zeros_like(x_image_true), np.zeros_like(y_image_true), x_image_true, y_image_true, 0.,
                                 self.zlens, self.kwargs_epl)

        npt.assert_almost_equal(x_fore, xtrue)
        npt.assert_almost_equal(y_fore, ytrue)

        args_lens = self.param_class.kwargs_to_args(self.kwargs_epl)
        xfast, yfast = fast_rayshooting.ray_shooting_fast(args_lens)

        x, y = self.lensModel.ray_shooting(x_image_true, y_image_true, self.kwargs_epl)

        npt.assert_almost_equal(xfast, x)
        npt.assert_almost_equal(yfast, y)

        x_inner, y_inner = fast_rayshooting.lensModel.ray_shooting(x_image_true, y_image_true, self.kwargs_epl)
        npt.assert_almost_equal(x_inner, xfast)
        npt.assert_almost_equal(y_inner, yfast)

        foreground_rays = fast_rayshooting._foreground_rays

        fast_rayshooting_new = MultiplaneFast(x_image_true, y_image_true, self.zlens, self.zsource,
                                           self.lens_model_list_epl, self.zlist_epl,
                 astropy_instance=None, param_class=self.param_class, foreground_rays=foreground_rays,
                 tol_source=1e-5, numerical_alpha_class=None)

        xfast, yfast = fast_rayshooting_new.ray_shooting_fast(args_lens)

        npt.assert_almost_equal(xfast, x)
        npt.assert_almost_equal(yfast, y)

        chi_square_source = fast_rayshooting.source_plane_chi_square(args_lens)
        chi_square_total = fast_rayshooting.chi_square(args_lens)

        logL = fast_rayshooting.logL(args_lens)
        logL_true = -0.5 * chi_square_total
        npt.assert_almost_equal(logL, logL_true)
        npt.assert_almost_equal(chi_square_total, chi_square_source)


if __name__ == '__main__':
    pytest.main()

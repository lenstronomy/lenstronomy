import pytest
import numpy as np
from lenstronomy.Util.param_util import ellipticity2phi_q
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.QuadOptimizer.param_manager import PowerLawFixedShear, \
    PowerLawFixedShearMultipole, PowerLawFreeShear, PowerLawFreeShearMultipole
from lenstronomy.LensModel.QuadOptimizer.optimizer import Optimizer
import numpy.testing as npt


class TestOptimizer(object):

    def setup(self):

        self.zlens, self.zsource = 0.5, 1.5
        epl_kwargs = {'theta_E': 1., 'center_x': 0., 'center_y': 0., 'e1': 0.2, 'e2': 0.1, 'gamma': 2.05}
        shear_kwargs = {'gamma1': 0.05, 'gamma2': -0.04}
        kwargs_macro = [epl_kwargs, shear_kwargs]

        self.x_image = np.array([0.65043538, -0.31109505, 0.78906059, -0.86222271])
        self.y_image = np.array([-0.89067493, 0.94851787,  0.52882605, -0.25403778])

        halo_list = ['SIS', 'SIS', 'SIS']
        halo_z = [self.zlens - 0.1, self.zlens, self.zlens + 0.4]
        halo_kwargs = [{'theta_E': 0.1, 'center_x': 0.3, 'center_y': -0.9},
                       {'theta_E': 0.15, 'center_x': 1.3, 'center_y': -0.5},
                       {'theta_E': 0.06, 'center_x': -0.4, 'center_y': -0.4}]

        self.kwargs_epl = kwargs_macro + halo_kwargs
        self.zlist_epl = [self.zlens, self.zlens] + halo_z
        self.lens_model_list_epl = ['EPL', 'SHEAR'] + halo_list

        kwargs_multi = [{'m': 4, 'a_m': -0.04, 'phi_m': -0.2, 'center_x': 0.1, 'center_y': -0.1}]
        self.kwargs_multipole = kwargs_macro + kwargs_multi + halo_kwargs
        self.zlist_multipole = [self.zlens, self.zlens, self.zlens] + halo_z
        self.lens_model_list_multipole = ['EPL', 'SHEAR'] + ['MULTIPOLE'] + halo_list


    def test_elp_free_shear(self):

        param_class = PowerLawFreeShear(self.kwargs_epl)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_epl, self.zlist_epl,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_epl, self.zlens, self.zsource, self.zlist_epl, multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

    def test_elp_fixed_shear(self):

        param_class = PowerLawFixedShear(self.kwargs_epl, 0.06)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_epl, self.zlist_epl,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_epl, self.zlens, self.zsource, self.zlist_epl, multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

        kwargs_shear = kwargs_final[1]
        shear_out = np.hypot(kwargs_shear['gamma1'], kwargs_shear['gamma2'])
        npt.assert_almost_equal(shear_out, 0.06)

    def test_multipole_free_shear(self):

        param_class = PowerLawFreeShearMultipole(self.kwargs_multipole)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_multipole, self.zlist_multipole,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_multipole, self.zlens, self.zsource, self.zlist_multipole, multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

        kwargs_epl = kwargs_final[0]
        kwargs_multipole = kwargs_final[2]
        npt.assert_almost_equal(kwargs_multipole['m'], 4)
        npt.assert_almost_equal(kwargs_multipole['center_x'], kwargs_epl['center_x'])
        npt.assert_almost_equal(kwargs_multipole['center_y'], kwargs_epl['center_y'])
        phi, _ = ellipticity2phi_q(kwargs_epl['e1'], kwargs_epl['e2'])
        npt.assert_almost_equal(phi, kwargs_multipole['phi_m'])

    def test_multipole_fixed_shear(self):

        param_class = PowerLawFixedShearMultipole(self.kwargs_multipole, 0.07)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_multipole, self.zlist_multipole,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_multipole, self.zlens, self.zsource, self.zlist_multipole, multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

        kwargs_shear = kwargs_final[1]
        shear_out = np.hypot(kwargs_shear['gamma1'], kwargs_shear['gamma2'])
        npt.assert_almost_equal(shear_out, 0.07)

        kwargs_epl = kwargs_final[0]
        kwargs_multipole = kwargs_final[2]
        npt.assert_almost_equal(kwargs_multipole['m'], 4)
        npt.assert_almost_equal(kwargs_multipole['center_x'], kwargs_epl['center_x'])
        npt.assert_almost_equal(kwargs_multipole['center_y'], kwargs_epl['center_y'])
        phi, _ = ellipticity2phi_q(kwargs_epl['e1'], kwargs_epl['e2'])
        npt.assert_almost_equal(phi, kwargs_multipole['phi_m'])

    def test_options(self):

        param_class = PowerLawFixedShearMultipole(self.kwargs_multipole, 0.07)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_multipole, self.zlist_multipole,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000, particle_swarm=False,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_multipole, self.zlens, self.zsource, self.zlist_multipole, multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

        kwargs_shear = kwargs_final[1]
        shear_out = np.hypot(kwargs_shear['gamma1'], kwargs_shear['gamma2'])
        npt.assert_almost_equal(shear_out, 0.07)

        foreground_rays = optimizer.fast_rayshooting._foreground_rays
        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_multipole, self.zlist_multipole,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000, particle_swarm=False,
                              re_optimize=True, re_optimize_scale=0.5,
                              foreground_rays=foreground_rays, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True)

        lensmodel = LensModel(self.lens_model_list_multipole, self.zlens, self.zsource, self.zlist_multipole,
                              multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

        kwargs_shear = kwargs_final[1]
        shear_out = np.hypot(kwargs_shear['gamma1'], kwargs_shear['gamma2'])
        npt.assert_almost_equal(shear_out, 0.07)

    def test_multi_threading(self):

        param_class = PowerLawFixedShearMultipole(self.kwargs_multipole, 0.07)

        optimizer = Optimizer(self.x_image, self.y_image, self.lens_model_list_multipole, self.zlist_multipole,
                              self.zlens, self.zsource, param_class, pso_convergence_mean=50000, particle_swarm=False,
                              foreground_rays=None, tol_source=1e-5, tol_simplex_func=1e-3, simplex_n_iterations=400)

        kwargs_final, source = optimizer.optimize(50, 100, verbose=True, threadCount=5)
        lensmodel = LensModel(self.lens_model_list_multipole, self.zlens, self.zsource, self.zlist_multipole,
                              multi_plane=True)
        beta_x, beta_y = lensmodel.ray_shooting(self.x_image, self.y_image, kwargs_final)

        npt.assert_almost_equal(np.sum(beta_x) - 4 * np.mean(beta_x), 0)
        npt.assert_almost_equal(np.sum(beta_y) - 4 * np.mean(beta_y), 0)

if __name__ == '__main__':
    pytest.main()

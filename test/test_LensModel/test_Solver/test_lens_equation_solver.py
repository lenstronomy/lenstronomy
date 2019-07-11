__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel


class TestLensEquationSolver(object):

    def setup(self):
        """

        :return:
        """
        pass

    def test_spep_sis(self):
        lens_model_list = ['SPEP', 'SIS']
        lensModel = LensModel(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.2, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_nfw(self):
        lens_model_list = ['NFW_ELLIPSE', 'SIS']
        lensModel = LensModel(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        Rs = 4.
        kwargs_lens = [{'alpha_Rs': 1., 'Rs': Rs, 'e1': 0.2, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     min_distance=min_distance,
                                                                     search_window=search_window,
                                                                     precision_limit=10**(-10), num_iter_max=10,
                                                                     verbose=True)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_foreground_shear(self):
        lens_model_list = ['SPEP', 'FOREGROUND_SHEAR']
        lensModel = LensModel(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.2, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1}, {'e1': 0.01, 'e2': -0.05}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_multiplane(self):
        lens_model_list = ['SPEP', 'SIS']
        lensModel = LensModel(lens_model_list, z_source=1., lens_redshift_list=[0.5, 0.3], multi_plane=True)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.2, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1}, {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_central_image(self):
        lens_model_list = ['SPEP', 'SIS', 'SHEAR']
        kwargs_spep = {'theta_E': 1, 'gamma': 2, 'e1': 0.2, 'e2': -0.03, 'center_x': 0, 'center_y': 0}
        kwargs_sis = {'theta_E': 1, 'center_x': 1.5, 'center_y': 0}
        kwargs_shear = {'e1': 0.01, 'e2': 0}
        kwargs_lens = [kwargs_spep, kwargs_sis, kwargs_shear]
        lensModel = LensModel(lens_model_list)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     min_distance=min_distance,
                                                                     search_window=search_window,
                                                                     precision_limit=10 ** (-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)
        print(x_pos, y_pos)
        assert len(x_pos) == 4

    def test_example(self):
        lens_model_list = ['SPEP', 'SHEAR']
        lensModel = LensModel(lens_model_list)

        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.03
        sourcePos_y = 0.0
        min_distance = 0.05
        search_window = 10
        gamma = 2.
        e1, e2 = -0.04, -0.1
        kwargs_shear = {'e1': e1, 'e2': e2}  # shear values to the source plane
        kwargs_spemd = {'theta_E': 1., 'gamma': gamma, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.01,
                        'e2': 0.05}  # parameters of the deflector lens model

        kwargs_lens = [kwargs_spemd, kwargs_shear]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     min_distance=min_distance,
                                                                     search_window=search_window,
                                                                     precision_limit=10 ** (-10), num_iter_max=10,
                                                                     arrival_time_sort=True)

        x_pos_non_linear, y_pos_non_linear = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     min_distance=min_distance,
                                                                     search_window=search_window,
                                                                     precision_limit=10 ** (-10), num_iter_max=10,
                                                                     arrival_time_sort=True, non_linear=True)

        x_pos_stoch, y_pos_stoch = lensEquationSolver.image_position_stochastic(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                                search_window=search_window,
                                                                                precision_limit=10 ** (-10),
                                                                                arrival_time_sort=True, x_center=0,
                                                                                y_center=0, num_random=100,
                                                                                )
        assert len(x_pos) == 4
        assert len(x_pos_stoch) == 4
        assert len(x_pos_non_linear) == 4
        npt.assert_almost_equal(x_pos, x_pos_stoch, decimal=5)
        npt.assert_almost_equal(x_pos, x_pos_non_linear, decimal=5)


if __name__ == '__main__':
    pytest.main()

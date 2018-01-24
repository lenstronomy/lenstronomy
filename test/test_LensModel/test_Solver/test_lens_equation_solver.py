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
        lensEquationSolver = LensEquationSolver(lens_model_list)
        lensModel = LensModel(lens_model_list)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_nfw(self):
        lens_model_list = ['NFW_ELLIPSE', 'SIS']
        lensModel = LensModel(lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_list)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        Rs = 4.
        kwargs_lens = [{'theta_Rs': 1., 'Rs': Rs,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)

    def test_foreground_shear(self):
        lens_model_list = ['SPEP', 'FOREGROUND_SHEAR']
        lensEquationSolver = LensEquationSolver(lens_model_list)
        lensModel = LensModel(lens_model_list)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        min_distance = 0.05
        search_window = 10
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}, {'e1': 0.01, 'e2': -0.05}]
        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=min_distance, search_window=search_window, precision_limit=10**(-10), num_iter_max=10)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)


if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.lens_model import LensModel


class TestSolver4Point(object):

    def setup(self):
        """

        :return:
        """
        pass

    def test_decoupling(self):
        lens_model_list = ['SPEP', 'SIS']
        solver = Solver4Point(lens_model_list=lens_model_list, decoupling=False)
        solver_decoupled = Solver4Point(lens_model_list=lens_model_list, decoupling=True)
        lensEquationSolver = LensEquationSolver(lens_model_list)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma,'q': 0.9, 'phi_G': 1.5, 'center_x': 0., 'center_y': 0}, {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        kwargs_lens_new = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        kwargs_lens_new_2 = solver_decoupled.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        print(kwargs_lens_new_2)
        print(kwargs_lens_new)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)

        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens_new_2[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens_new_2[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens_new_2[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens_new_2[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens_new_2[0]['center_y'], decimal=3)

        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], 1., decimal=3)
        lensModel = LensModel(lens_model_list=lens_model_list)
        x_source_new, y_source_new = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        dist = np.sqrt((x_source_new - x_source_new[0]) ** 2 + (y_source_new - y_source_new[0]) ** 2)
        print(dist)
        assert np.max(dist) < 0.000001

    def test_solver_spep(self):
        lens_model_list = ['SPEP']
        solver = Solver4Point(lens_model_list=lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_list)

        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma,'q': 0.9, 'phi_G': 1.5, 'center_x': 0., 'center_y': 0}]
        kwargs_lens_new = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], 1., decimal=3)
        lensModel = LensModel(lens_model_list=lens_model_list)
        x_source_new, y_source_new = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        dist = np.sqrt((x_source_new - x_source_new[0]) ** 2 + (y_source_new - y_source_new[0]) ** 2)
        print(dist)
        assert np.max(dist) < 0.000001

    def test_solver_nfw(self):
        lens_model_list = ['NFW_ELLIPSE', 'SIS']
        solver = Solver4Point(lens_model_list=lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_list)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        Rs = 4.
        kwargs_lens = [{'theta_Rs': 1., 'Rs': Rs,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        kwargs_lens_init = [{'theta_Rs': 0.5, 'Rs': Rs,'q': 0.9, 'phi_G': 1.5, 'center_x': 0., 'center_y': 0}, kwargs_lens[1]]
        kwargs_lens_new = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_Rs'], kwargs_lens[0]['theta_Rs'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)


if __name__ == '__main__':
    pytest.main()
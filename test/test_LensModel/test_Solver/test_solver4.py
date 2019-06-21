__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.solver4point import Solver4Point
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.param_util as param_util


class TestSolver4Point(object):

    def setup(self):
        """

        :return:
        """
        pass

    def test_decoupling(self):
        lens_model_list = ['SPEP', 'SIS']
        lensModel = LensModel(lens_model_list)
        solver = Solver4Point(lensModel)
        solver_decoupled = Solver4Point(lensModel)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        phi_G, q = 1.5, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0}, {'theta_E': 0.1, 'center_x': 0.5, 'center_y': 0}]
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        kwargs_lens_new_2, accuracy = solver_decoupled.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        print(kwargs_lens_new_2)
        print(kwargs_lens_new)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e1'], kwargs_lens[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e2'], kwargs_lens[0]['e2'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)

        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens_new_2[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e1'], kwargs_lens_new_2[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e2'], kwargs_lens_new_2[0]['e2'], decimal=3)
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
        lensModel = LensModel(lens_model_list)
        solver = Solver4Point(lensModel)
        lensEquationSolver = LensEquationSolver(lensModel)

        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        phi_G, q = 1.5, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0}]
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        npt.assert_almost_equal(kwargs_lens_new[0]['theta_E'], kwargs_lens[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e1'], kwargs_lens[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e2'], kwargs_lens[0]['e2'], decimal=3)
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
        lensModel = LensModel(lens_model_list)
        solver = Solver4Point(lensModel)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        Rs = 4.
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [{'alpha_Rs': 1., 'Rs': Rs, 'e1': e1, 'e2': e2, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        phi_G, q = 1.5, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens_init = [{'alpha_Rs': 0.5, 'Rs': Rs, 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0}, kwargs_lens[1]]
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        npt.assert_almost_equal(kwargs_lens_new[0]['alpha_Rs'], kwargs_lens[0]['alpha_Rs'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e1'], kwargs_lens[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['e2'], kwargs_lens[0]['e2'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)

    def test_solver_shapelets(self):
        lens_model_list = ['SHAPELETS_CART', 'SPEP']
        lensModel = LensModel(lens_model_list)
        solver = Solver4Point(lensModel)
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.
        deltapix = 0.05
        numPix = 150
        coeffs = np.array([0, 0.1, 0.1, 0, 0, -0.1])
        kwargs_lens = [{'beta': 1., 'coeffs': coeffs, 'center_x': 0., 'center_y': 0.},
                       {'theta_E': 1., 'gamma': 2, 'e1': 0.1, 'e2': 0, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        print(x_pos, y_pos)
        kwargs_lens_init = [{'beta': 1, 'coeffs': np.zeros_like(coeffs), 'center_x': 0., 'center_y': 0}, kwargs_lens[1]]
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        npt.assert_almost_equal(kwargs_lens_new[0]['beta'], kwargs_lens[0]['beta'], decimal=3)
        coeffs_new = kwargs_lens_new[0]['coeffs']
        for i in range(len(coeffs)):
            npt.assert_almost_equal(coeffs_new[i], coeffs[i], decimal=3)

    def test_solver_simplified(self):
        lens_model_list = ['SPEP', 'SHEAR']
        lensModel = LensModel(lens_model_list)

        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        gamma_ext = 0.05
        e1, e2 = param_util.phi_gamma_ellipticity(phi=0.4, gamma=gamma_ext)
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.1, 'e2': -0.1, 'center_x': 0.1, 'center_y': -0.1},
                       {'e1': e1, 'e2': e2}]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        e1_new, e2_new = param_util.phi_gamma_ellipticity(phi=0., gamma=gamma_ext+0.1)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0},
                            {'e1': e1_new, 'e2': e2_new}]
        solver = Solver4Point(lensModel, solver_type='PROFILE_SHEAR')
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        assert accuracy < 10**(-10)
        x_source, y_source = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        x_source, y_source = np.mean(x_source), np.mean(y_source)
        x_pos_new, y_pos_new = lensEquationSolver.findBrightImage(x_source, y_source, kwargs_lens_new, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, x_pos_new)
        x_pos = np.sort(x_pos)
        x_pos_new = np.sort(x_pos_new)
        y_pos = np.sort(y_pos)
        y_pos_new = np.sort(y_pos_new)
        for i in range(len(x_pos)):
            npt.assert_almost_equal(x_pos[i], x_pos_new[i], decimal=6)
            npt.assert_almost_equal(y_pos[i], y_pos_new[i], decimal=6)

    def test_solver_simplified_2(self):
        lens_model_list = ['SPEP', 'SHEAR']
        lensModel = LensModel(lens_model_list)

        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.96
        e1, e2 = -0.01, -0.01
        kwargs_shear = {'e1': e1, 'e2': e2}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        kwargs_spemd = {'theta_E': 1., 'gamma': gamma, 'center_x': 0, 'center_y': 0, 'e1': -0.2, 'e2': -0.03}
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': 0, 'e2': 0, 'center_x': 0., 'center_y': 0},
                            {'e1': e1, 'e2': e2}]
        solver = Solver4Point(lensModel, solver_type='PROFILE_SHEAR')
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        assert accuracy < 10**(-10)
        x_source, y_source = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        x_source, y_source = np.mean(x_source), np.mean(y_source)
        x_pos_new, y_pos_new = lensEquationSolver.findBrightImage(x_source, y_source, kwargs_lens_new, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, x_pos_new)
        x_pos = np.sort(x_pos)
        x_pos_new = np.sort(x_pos_new)
        y_pos = np.sort(y_pos)
        y_pos_new = np.sort(y_pos_new)
        for i in range(len(x_pos)):
            npt.assert_almost_equal(x_pos[i], x_pos_new[i], decimal=6)
            npt.assert_almost_equal(y_pos[i], y_pos_new[i], decimal=6)
        npt.assert_almost_equal(kwargs_lens_new[1]['e1'], kwargs_lens[1]['e1'], decimal=8)
        npt.assert_almost_equal(kwargs_lens_new[1]['e2'], kwargs_lens[1]['e2'], decimal=8)

    def test_solver_profile_shear(self):
        lens_model_list = ['SPEP', 'SHEAR']
        lensModel = LensModel(lens_model_list)

        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.
        sourcePos_y = 0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.98
        e1, e2 = -0.04, -0.01
        gamma_ext = np.sqrt(e1**2 + e2**2)
        kwargs_shear = {'e1': e1, 'e2': e2}  # shear values to the source plane
        kwargs_spemd = {'theta_E': 1.66, 'gamma': gamma, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.1,
                        'e2': 0.05}  # parameters of the deflector lens model

        kwargs_lens = [kwargs_spemd, kwargs_shear]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, y_pos, 'test positions')
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': 0, 'e2': 0, 'center_x': 0., 'center_y': 0},
                            {'e1': e1, 'e2': -e2}]
        solver = Solver4Point(lensModel, solver_type='PROFILE_SHEAR')
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        assert accuracy < 10**(-10)
        x_source, y_source = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        x_source, y_source = np.mean(x_source), np.mean(y_source)
        x_pos_new, y_pos_new = lensEquationSolver.findBrightImage(x_source, y_source, kwargs_lens_new, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, x_pos_new)
        x_pos = np.sort(x_pos)
        x_pos_new = np.sort(x_pos_new)
        y_pos = np.sort(y_pos)
        y_pos_new = np.sort(y_pos_new)
        for i in range(len(x_pos)):
            npt.assert_almost_equal(x_pos[i], x_pos_new[i], decimal=6)
            npt.assert_almost_equal(y_pos[i], y_pos_new[i], decimal=6)
        npt.assert_almost_equal(kwargs_lens_new[1]['e1'], kwargs_lens[1]['e1'], decimal=8)
        npt.assert_almost_equal(kwargs_lens_new[1]['e2'], kwargs_lens[1]['e2'], decimal=8)
        npt.assert_almost_equal(kwargs_lens_new[0]['e1'], kwargs_lens[0]['e1'], decimal=8)
        npt.assert_almost_equal(kwargs_lens_new[0]['e2'], kwargs_lens[0]['e2'], decimal=8)

    def test_solver_multiplane(self):
        lens_model_list = ['SPEP', 'SHEAR', 'SIS']
        lensModel = LensModel(lens_model_list, z_source=1, lens_redshift_list=[0.5, 0.5, 0.3], multi_plane=True)

        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.01
        sourcePos_y = -0.01
        deltapix = 0.05
        numPix = 150
        gamma = 1.96
        e1, e2 = 0.01, 0.01
        kwargs_shear = {'e1': e1, 'e2': e2}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        kwargs_spemd = {'theta_E': 1., 'gamma': gamma, 'center_x': 0, 'center_y': 0, 'e1': 0.2, 'e2': 0.03}
        kwargs_sis = {'theta_E': .1, 'center_x': 1, 'center_y': 0}
        kwargs_lens = [kwargs_spemd, kwargs_shear, kwargs_sis]
        x_pos, y_pos = lensEquationSolver.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, y_pos, 'test positions')
        kwargs_lens_init = [{'theta_E': 1.3, 'gamma': gamma, 'e1': 0.1, 'e2': 0, 'center_x': 0., 'center_y': 0},
                            {'e1': e1, 'e2': e2}, {'theta_E': .1, 'center_x': 1, 'center_y': 0}]
        solver = Solver4Point(lensModel, solver_type='PROFILE')
        kwargs_lens_new, accuracy = solver.constraint_lensmodel(x_pos, y_pos, kwargs_lens_init)
        print(kwargs_lens_new, 'kwargs_lens_new')
        assert accuracy < 10**(-10)
        x_source, y_source = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        x_source, y_source = np.mean(x_source), np.mean(y_source)
        x_pos_new, y_pos_new = lensEquationSolver.findBrightImage(x_source, y_source, kwargs_lens_new, numImages=4,
                                                          min_distance=deltapix, search_window=numPix * deltapix)
        print(x_pos, x_pos_new)
        x_pos = np.sort(x_pos)
        x_pos_new = np.sort(x_pos_new)
        y_pos = np.sort(y_pos)
        y_pos_new = np.sort(y_pos_new)
        for i in range(len(x_pos)):
            npt.assert_almost_equal(x_pos[i], x_pos_new[i], decimal=6)
            npt.assert_almost_equal(y_pos[i], y_pos_new[i], decimal=6)
        npt.assert_almost_equal(kwargs_lens_new[1]['e1'], kwargs_lens[1]['e1'], decimal=8)


if __name__ == '__main__':
    pytest.main()

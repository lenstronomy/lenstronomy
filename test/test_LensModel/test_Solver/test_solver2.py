__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Solver.solver2point import Solver2Point
import lenstronomy.Util.param_util as param_util


class TestSolver(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        pass

    def test_subtract(self):
        lensModel = LensModel(['SPEP'])
        solver_spep_center = Solver2Point(lensModel, solver_type='CENTER')
        x_cat = np.array([0, 0])
        y_cat = np.array([1, 2])
        a = solver_spep_center._subtract_constraint(x_cat, y_cat)
        assert a[0] == 0
        assert a[1] == 1

    def test_all_spep(self):
        lensModel = LensModel(['SPEP'])
        solver_spep_center = Solver2Point(lensModel, solver_type='CENTER')
        solver_spep_ellipse = Solver2Point(lensModel, solver_type='ELLIPSE')
        image_position_spep = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        gamma = 1.9
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [{'theta_E': 1, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = image_position_spep.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=2, min_distance=0.01, search_window=5, precision_limit=10**(-10), num_iter_max=10)
        print(x_pos, y_pos, 'test')
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        kwargs_init = [{'theta_E': 1, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        kwargs_out_center, precision = solver_spep_center.constraint_lensmodel(x_pos, y_pos, kwargs_init)

        kwargs_init = [{'theta_E': 1, 'gamma': gamma, 'e1': 0, 'e2': 0, 'center_x': 0.1, 'center_y': -0.1}]
        kwargs_out_ellipse, precision = solver_spep_ellipse.constraint_lensmodel(x_pos, y_pos, kwargs_init)

        npt.assert_almost_equal(kwargs_out_center[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], -0.1, decimal=3)

        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], kwargs_lens[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e2'], kwargs_lens[0]['e2'], decimal=3)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], e1, decimal=3)

    def test_all_nfw(self):
        lensModel = LensModel(['SPEP'])
        solver_nfw_ellipse = Solver2Point(lensModel, solver_type='ELLIPSE')
        solver_nfw_center = Solver2Point(lensModel, solver_type='CENTER')
        spep = LensModel(['SPEP'])

        image_position_nfw = LensEquationSolver(LensModel(['SPEP', 'NFW']))
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        Rs = 0.1
        nfw = NFW()
        theta_Rs = nfw._rho02alpha(1., Rs)
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0.1, 'center_y': -0.1},
                       {'Rs': Rs, 'theta_Rs': theta_Rs, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = image_position_nfw.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=2, min_distance=deltapix, search_window=numPix*deltapix)
        print(len(x_pos), 'number of images')
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]

        kwargs_init = [{'theta_E': 1, 'gamma': gamma, 'e1': e1, 'e2': e2, 'center_x': 0., 'center_y': 0},
                       {'Rs': Rs, 'theta_Rs': theta_Rs, 'center_x': -0.5, 'center_y': 0.5}]
        kwargs_out_center, precision = solver_nfw_center.constraint_lensmodel(x_pos, y_pos, kwargs_init)
        source_x, source_y = spep.ray_shooting(x_pos[0], y_pos[0], kwargs_out_center)
        x_pos_new, y_pos_new = image_position_nfw.findBrightImage(source_x, source_y, kwargs_out_center, numImages=2, min_distance=deltapix, search_window=numPix*deltapix)
        print(kwargs_out_center, 'kwargs_out_center')
        npt.assert_almost_equal(x_pos_new[0], x_pos[0], decimal=2)
        npt.assert_almost_equal(y_pos_new[0], y_pos[0], decimal=2)

        npt.assert_almost_equal(kwargs_out_center[0]['center_x'], kwargs_lens[0]['center_x'], decimal=2)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], kwargs_lens[0]['center_y'], decimal=2)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], -0.1, decimal=2)

        kwargs_init = [{'theta_E': 1., 'gamma': gamma, 'e1': 0, 'e2': 0, 'center_x': 0.1, 'center_y': -0.1},
                       {'Rs': Rs, 'theta_Rs': theta_Rs, 'center_x': -0.5, 'center_y': 0.5}]
        kwargs_out_ellipse, precision = solver_nfw_ellipse.constraint_lensmodel(x_pos, y_pos, kwargs_init)

        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], kwargs_lens[0]['e1'], decimal=2)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e2'], kwargs_lens[0]['e2'], decimal=2)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], e1, decimal=2)

    def test_all_spep_sis(self):
        lensModel = LensModel(['SPEP', 'SIS'])
        solver_ellipse = Solver2Point(lensModel, solver_type='ELLIPSE')
        solver_center = Solver2Point(lensModel, solver_type='CENTER')
        spep = LensModel(['SPEP', 'SIS'])
        image_position = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.2, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.6, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = image_position.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=2, min_distance=deltapix, search_window=numPix*deltapix, precision_limit=10**(-10))
        print(len(x_pos), 'number of images')
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]

        kwargs_init = [{'theta_E': 1, 'gamma': gamma, 'e1': 0.2, 'e2': -0.03, 'center_x': 0., 'center_y': 0},
                       {'theta_E': 0.6, 'center_x': -0.5, 'center_y': 0.5}]
        kwargs_out_center, precision = solver_center.constraint_lensmodel(x_pos, y_pos, kwargs_init)
        print(kwargs_out_center, 'output')
        source_x, source_y = spep.ray_shooting(x_pos[0], y_pos[0], kwargs_out_center)
        x_pos_new, y_pos_new = image_position.findBrightImage(source_x, source_y, kwargs_out_center, numImages=2, min_distance=deltapix, search_window=numPix*deltapix)
        npt.assert_almost_equal(x_pos_new[0], x_pos[0], decimal=3)
        npt.assert_almost_equal(y_pos_new[0], y_pos[0], decimal=3)

        npt.assert_almost_equal(kwargs_out_center[0]['center_x'], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], kwargs_lens[0]['center_y'], decimal=3)
        npt.assert_almost_equal(kwargs_out_center[0]['center_y'], -0.1, decimal=3)

        kwargs_init = [{'theta_E': 1., 'gamma': gamma, 'e1': 0, 'e2': 0, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.6, 'center_x': -0.5, 'center_y': 0.5}]
        kwargs_out_ellipse, precision = solver_ellipse.constraint_lensmodel(x_pos, y_pos, kwargs_init)

        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], kwargs_lens[0]['e1'], decimal=3)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e2'], kwargs_lens[0]['e2'], decimal=3)
        npt.assert_almost_equal(kwargs_out_ellipse[0]['e1'], 0.2, decimal=3)

    def test_shapelet_cart(self):
        lens_model_list = ['SHAPELETS_CART', 'SIS']
        lens = LensModel(lens_model_list)
        solver = Solver2Point(lens, solver_type='SHAPELETS')
        image_position = LensEquationSolver(lens)
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100

        kwargs_lens = [{'coeffs': [1., 0., 0.1, 1.], 'beta': 1.},
                       {'theta_E': 1., 'center_x': -0.1, 'center_y': 0.1}]
        x_pos, y_pos = image_position.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=2, min_distance=deltapix, search_window=numPix*deltapix, precision_limit=10**(-10))
        print(len(x_pos), 'number of images')
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]

        kwargs_init = [{'coeffs': [1., 0., 0.1, 1.], 'beta': 1.},
                       {'theta_E': 1., 'center_x': -0.1, 'center_y': 0.1}]
        kwargs_out, precision = solver.constraint_lensmodel(x_pos, y_pos, kwargs_init)
        print(kwargs_out, 'output')
        source_x, source_y = lens.ray_shooting(x_pos[0], y_pos[0], kwargs_out)
        x_pos_new, y_pos_new = image_position.findBrightImage(source_x, source_y, kwargs_out, numImages=2, min_distance=deltapix, search_window=numPix*deltapix)
        npt.assert_almost_equal(x_pos_new[0], x_pos[0], decimal=3)
        npt.assert_almost_equal(y_pos_new[0], y_pos[0], decimal=3)

        npt.assert_almost_equal(kwargs_out[0]['coeffs'][1], kwargs_lens[0]['coeffs'][1], decimal=3)
        npt.assert_almost_equal(kwargs_out[0]['coeffs'][2], kwargs_lens[0]['coeffs'][2], decimal=3)

    def test_theta_E_phi(self):
        lensModel = LensModel(['SPEP', 'SHEAR'])
        solver = Solver2Point(lensModel, solver_type='THETA_E_PHI')

        image_position = LensEquationSolver(lensModel)
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'e1': 0.1, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1},
                       {'e1': 0.03, 'e2': 0.0}]
        x_pos, y_pos = image_position.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, numImages=2, min_distance=deltapix, search_window=numPix*deltapix, precision_limit=10**(-15))
        print(len(x_pos), 'number of images')
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]

        kwargs_init = [{'theta_E': 1.9, 'gamma': gamma, 'e1': 0.1, 'e2': -0.03, 'center_x': 0.1, 'center_y': -0.1},
                       {'e1': 0., 'e2': 0.03}]
        kwargs_out, precision = solver.constraint_lensmodel(x_pos, y_pos, kwargs_init)
        print(kwargs_out, 'output')
        source_x, source_y = lensModel.ray_shooting(x_pos[0], y_pos[0], kwargs_out)
        x_pos_new, y_pos_new = image_position.findBrightImage(source_x, source_y, kwargs_out, numImages=2, min_distance=deltapix, search_window=numPix*deltapix)
        npt.assert_almost_equal(x_pos_new[0], x_pos[0], decimal=3)
        npt.assert_almost_equal(y_pos_new[0], y_pos[0], decimal=3)

        npt.assert_almost_equal(kwargs_out[0]['theta_E'], kwargs_lens[0]['theta_E'], decimal=3)
        npt.assert_almost_equal(kwargs_out[1]['e1'], kwargs_lens[1]['e1'], decimal=2)
        npt.assert_almost_equal(kwargs_out[1]['e2'], kwargs_lens[1]['e2'], decimal=2)


if __name__ == '__main__':
    pytest.main()

__author__ = 'sibirrer'

import astrofunc.util as util
import numpy as np
import numpy.testing as npt
import pytest
from astrofunc.LensingProfiles.nfw import NFW
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.Solver.image_positions import LensEquationSolver
from lenstronomy.Solver.solver4point import SolverProfile, Constraints, SolverShapelets
from lenstronomy.Solver.solver4point_full import Solver4Point


class TestSolver(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.constraints = Constraints(lens_model='SPEP')
        self.lens_spep = LensModel(['SPEP'])
        self.Image_spep = LensEquationSolver(['SPEP'])
        self.lens_nfw = LensModel(['SPEP', 'NFW'])
        self.Image_nfw = LensEquationSolver(['SPEP', 'NFW'])
        self.lens_spp = LensModel(['SPEP', 'SPP'])
        self.Image_spp = LensEquationSolver(['SPEP', 'SPP'])
        self.solver = SolverProfile()
        self.nfw = NFW()

    def test_subtract(self):
        x_cat = np.array([0, 0, 1, 2])
        y_cat = np.array([1, 2, 3, 2])
        x_sub = np.array([0, 2, 1, 1])
        y_sub = np.array([-1, 0, -1, -1])
        a = self.constraints._subtract_constraint(x_cat, y_cat, x_sub, y_sub)
        assert a[0] == 2

    def test_all_spep(self):
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1, 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = self.Image_spep.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        sourcePos_x, sourcePos_y = self.lens_spep.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x, sourcePos_y, 'source positions')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])

        init = np.array([2., 0, 0, 0., 0.1, 0])
        x_true = np.array([1., e1, e2, 0.1, -0.1, 0])
        kwargs_lens[0]['theta_E'] = 0

        x_sub, y_sub = self.lens_spep.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        print(a, 'a')
        print(self.solver.F(x_true, x_pos, y_pos, a, gamma), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma})
        x_ = self.solver.F(x, x_pos, y_pos, a, gamma)

        [phi_E, e1_new, e2_new, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1_new, e2_new)
        kwargs_lens_new = [{'theta_E': phi_E, 'gamma': gamma, 'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y}]
        sourcePos_x_new, sourcePos_y_new = self.lens_spep.ray_shooting(x_pos[0], y_pos[0], kwargs_lens_new)
        x_pos_new, y_pos_new = self.Image_spep.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        print(x_pos_new, 'x_pos_new')
        print(x_pos, 'x_pos old')
        print(kwargs_lens_new)
        npt.assert_almost_equal(x[0], 1, decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(x[3], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(x[4], kwargs_lens[0]['center_y'], decimal=3)

        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)
        npt.assert_almost_equal(x_[2], 0, decimal=3)
        npt.assert_almost_equal(x_[3], 0, decimal=3)
        npt.assert_almost_equal(x_[4], 0, decimal=3)

    def test_large_separations(self):
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.5
        numPix = 100
        gamma = 1.9
        kwargs_lens = [{'theta_E': 15., 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = self.Image_spep.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=deltapix, search_window=numPix*deltapix)
        sourcePos_x, sourcePos_y = self.lens_spep.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x, sourcePos_y, 'source positions')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])

        init = np.array([2., 0, 0, 0., 0.1, 0])
        x_true = np.array([1., e1, e2, 0.1, -0.1, 0])
        kwargs_lens[0]['theta_E'] = 0

        x_sub, y_sub = self.lens_spep.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        print(a, 'a')
        print(self.solver.F(x_true, x_pos, y_pos, a, gamma), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma})
        x_ = self.solver.F(x, x_pos, y_pos, a, gamma)

        [phi_E, e1_new, e2_new, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1_new, e2_new)
        kwargs_lens_new = [{'theta_E': phi_E, 'gamma': gamma, 'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y}]
        sourcePos_x_new, sourcePos_y_new = self.lens_spep.ray_shooting(x_pos[0], y_pos[0], kwargs_lens_new)
        x_pos_new, y_pos_new = self.Image_spep.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, min_distance=deltapix, search_window=numPix*deltapix)
        print(x_pos_new, 'x_pos_new')
        print(x_pos, 'x_pos old')
        print(kwargs_lens_new)
        npt.assert_almost_equal(x[0], 15, decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(x[3], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(x[4], kwargs_lens[0]['center_y'], decimal=3)

        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)
        npt.assert_almost_equal(x_[2], 0, decimal=3)
        npt.assert_almost_equal(x_[3], 0, decimal=3)
        npt.assert_almost_equal(x_[4], 0, decimal=3)

    def test_all_nfw(self):
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        Rs = 0.1
        theta_Rs = self.nfw._rho02alpha(1., Rs)
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'Rs': Rs, 'theta_Rs': theta_Rs, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = self.Image_nfw.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=deltapix, search_window=numPix*deltapix)
        sourcePos_x_new, sourcePos_y_new = self.lens_nfw.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x - sourcePos_x_new, 'sourcePos_x- sourcePos_x_new NFW')
        print(sourcePos_y - sourcePos_y_new, 'sourcePos_y- sourcePos_y_new NFW')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        init = np.array([1., 0, 0, 0., 0.05, -0.01])
        x_true = np.array([1., e1, e2, 0.1, -0.1, 0])
        # [Rs, rho0, r200, center_x_nfw, center_y_nfw] = param
        param = np.array([0.1, 1., 10., -0.5, +0.5])
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_nfw.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)

        print(self.solver.F(x_true, x_pos, y_pos, a, gamma), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma})
        x_ = self.solver.F(x, x_pos, y_pos, a, gamma)
        print(x, 'theta_E, e1, e2, center_x, center_y, non_sens')

        [phi_E, e1_new, e2_new, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1_new, e2_new)
        kwargs_lens_new = [{'theta_E': phi_E, 'gamma': gamma,'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y},
                           {'Rs': 0.1, 'theta_Rs': 1., 'center_x': -0.5, 'center_y': 0.5}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_nfw.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.Image_nfw.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, min_distance=deltapix, search_window=numPix*deltapix)
        # plt.plot(x_pos, y_pos, 'or')
        # plt.plot(x_pos_new, y_pos_new, 'og')
        # plt.show()
        print(x_pos_new-x_pos, 'x_pos_new - x_pos')
        npt.assert_almost_equal(x[0], 1, decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=3)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=3)
        npt.assert_almost_equal(x[3], kwargs_lens[0]['center_x'], decimal=3)
        npt.assert_almost_equal(x[4], kwargs_lens[0]['center_y'], decimal=3)

        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)
        npt.assert_almost_equal(x_[2], 0, decimal=3)
        npt.assert_almost_equal(x_[3], 0, decimal=3)
        npt.assert_almost_equal(x_[4], 0, decimal=3)
        npt.assert_almost_equal(x_[5], 0, decimal=3)

    def test_all_spp(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = self.Image_spp.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, min_distance=deltapix, search_window=numPix*deltapix, numImages=4)
        x_mapped, y_mapped = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        print(sourcePos_x - center_x, 'sourcePos_x- sourcePos_x_new SPP')
        print(sourcePos_y - center_y, 'sourcePos_y- sourcePos_y_new SPP')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        init = np.array([1., 0., 0., 0.05, -0.01, 0])
        # init = np.array([1., 1.9, 0.8, 0.5, 0.1, -0.1])
        x_true = np.array([1., e1, e2, 0.1, -0.1, 0])
        # [phi_E, gamma_spp, center_x_spp, center_y_spp] = param
        param = np.array([0.1, 1.9, -0.5, +0.5])
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_spp.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)

        print(self.solver.F(x_true, x_pos, y_pos, a, gamma), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma})
        x_ = self.solver.F(x, x_pos, y_pos, a, gamma)
        print(x, 'theta_E, gamma, e1, e2, center_x, center_y')

        [phi_E, e1_new, e2_new, center_x, center_y, no_sens_param] = x
        phi_G, q = util.elliptisity2phi_q(e1_new, e2_new)
        kwargs_lens_new = [{'theta_E': phi_E, 'gamma': gamma,'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y},
                           {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.Image_spp.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, min_distance=deltapix, search_window=numPix*deltapix, numImages=4)
        # import matplotlib.pylab as plt
        # plt.plot(x_pos, y_pos, 'or')
        # plt.plot(x_pos_new, y_pos_new, 'og')
        # plt.show()
        print(x_pos_new-x_pos, 'x_pos_new - x_pos')
        npt.assert_almost_equal(x[0], 1, decimal=2)
        npt.assert_almost_equal(kwargs_lens_new[0]['q'], kwargs_lens[0]['q'], decimal=2)
        npt.assert_almost_equal(kwargs_lens_new[0]['phi_G'], kwargs_lens[0]['phi_G'], decimal=2)
        npt.assert_almost_equal(x[3], kwargs_lens[0]['center_x'], decimal=2)
        npt.assert_almost_equal(x[4], kwargs_lens[0]['center_y'], decimal=2)

        npt.assert_almost_equal(x_[0], 0, decimal=2)
        npt.assert_almost_equal(x_[1], 0, decimal=2)
        npt.assert_almost_equal(x_[2], 0, decimal=2)
        npt.assert_almost_equal(x_[3], 0, decimal=2)
        npt.assert_almost_equal(x_[4], 0, decimal=2)
        npt.assert_almost_equal(x_[5], 0, decimal=2)


class TestSolverNew(object):

    def setup(self):
        self.lens_spep = LensModel(['SPEP'])
        self.lens_spep_spp = LensModel(['SPEP', 'SPP'])
        self.image_position_spep_spp = LensEquationSolver(['SPEP', 'SPP'])
        self.lens_spep_spp_shapelets = LensModel(['SPEP', 'SPP', 'SHAPELETS_CART'])
        self.image_position_spep_spp_shapelets = LensEquationSolver(['SPEP', 'SPP', 'SHAPELETS_CART'])
        self.solverShapelets = SolverShapelets()
        self.solver = SolverProfile()
        self.constraints = Constraints(lens_model='SHAPELETS_CART')

    def test_all_spp(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        beta = 1.5
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5},
                       {'coeffs': [0.,-0.1,0.01,-0.03,0.04, 0.1], 'beta': beta, 'center_x': 0, 'center_y': 0}]
        x_pos, y_pos = self.image_position_spep_spp_shapelets.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        x_mapped, y_mapped = self.lens_spep_spp_shapelets.ray_shooting(x_pos, y_pos, kwargs_lens)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        init = np.array([0, 0, 0, 0, 0, 0])
        x_true = np.array(kwargs_lens[2]['coeffs'])
        # [phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp] = spep_spp_param
        param = np.array([1., 1.9, 0.8, 0.5, 0.1, -0.1, 0.1, 1.9, -0.5, 0.5])
        kwargs_lens[2]['coeffs'] = [0, 0, 0, 0, 0, 0]
        x_sub, y_sub = self.lens_spep_spp_shapelets.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        kwargs = {'beta': kwargs_lens[2]['beta'], 'center_x': kwargs_lens[2]['center_x'], 'center_y': kwargs_lens[2]['center_y']}
        print(self.solverShapelets.F(x_true, x_pos, y_pos, a, **kwargs), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, kwargs)
        x_ = self.solverShapelets.F(x, x_pos, y_pos, a, **kwargs)
        x[0] = 0
        print(x, 'coeffs')

        #[phi_E, q, phi_G, center_x, center_y, no_sens_param] = x
        kwargs_lens_new = [{'theta_E': 1., 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                           {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5},
                           {'coeffs': x, 'beta': beta, 'center_x': 0, 'center_y': 0}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_spep_spp_shapelets.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.image_position_spep_spp_shapelets.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, kwargs_else=None, numImages=4, min_distance=deltapix, search_window=numPix*deltapix)
        # import matplotlib.pylab as plt
        # plt.plot(x_pos, y_pos, 'or')
        # plt.plot(x_pos_new, y_pos_new, 'og')
        # plt.show()
        print(x_pos_new-x_pos, 'x_pos_new - x_pos')
        #npt.assert_almost_equal(x[0], kwargs_lens['coeffs'][0], decimal=3)
        npt.assert_almost_equal(x[1], -0.1, decimal=3)
        npt.assert_almost_equal(x[2], 0.01, decimal=3)
        npt.assert_almost_equal(x[3], -0.03, decimal=3)
        npt.assert_almost_equal(x[4], 0.04, decimal=3)
        #npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)
        npt.assert_almost_equal(x_[2], 0, decimal=3)
        npt.assert_almost_equal(x_[3], 0, decimal=3)
        npt.assert_almost_equal(x_[4], 0, decimal=3)
        npt.assert_almost_equal(x_[5], 0, decimal=3)


class TestSolver4Point_full(object):

    def setup(self):
        """

        :return:
        """
        pass

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

    def test_solver_nfw(self):
        lens_model_list = ['NFW_ELLIPSE', 'SIS']
        solver = Solver4Point(lens_model_list=lens_model_list)
        lensModel = LensModel(lens_model_list)
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
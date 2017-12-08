__author__ = 'sibirrer'

import astrofunc.util as util
import numpy as np
import numpy.testing as npt
import pytest
from astrofunc.LensingProfiles.nfw import NFW
from lenstronomy.ImSim.lens_model import LensModel
from lenstronomy.Solver.image_positions import ImagePosition
from lenstronomy.Solver.solver2point import SolverCenter2, Constraints2, SolverShapelets2
from lenstronomy.Trash.solver2point_new import Constraints2_new
from lenstronomy.Trash.solver2point_new import SolverSPEP2_ellipse


class TestSolver(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.constraints = Constraints2(solver_type='CENTER', lens_model='SPEP')
        kwargs_options_spep = {'lens_model_list': ['SPEP']}
        self.lens_spep = LensModel(kwargs_options_spep)
        self.image_position_spep = ImagePosition(self.lens_spep)
        kwargs_options_nfw = {'lens_model_list': ['SPEP', 'NFW']}
        self.lens_nfw = LensModel(kwargs_options_nfw)
        self.image_position_nfw = ImagePosition(self.lens_nfw)
        kwargs_options_spp = {'lens_model_list': ['SPEP', 'SPP']}
        self.lens_spp = LensModel(kwargs_options_spp)
        self.image_position_spp = ImagePosition(self.lens_spp)
        self.solver = SolverCenter2()
        self.nfw = NFW()

    def test_subtract(self):
        x_cat = np.array([0, 0])
        y_cat = np.array([1, 2])
        x_sub = np.array([0, 2])
        y_sub = np.array([-1, 0])
        a = self.constraints._subtract_constraint(x_cat, y_cat, x_sub, y_sub)
        assert a[0] == 2

    def test_all_spep(self):
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1, 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = self.image_position_spep.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        sourcePos_x, sourcePos_y = self.lens_spep.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x, sourcePos_y, 'source positions')
        sourcePos_x = sourcePos_x[:2]
        sourcePos_y = sourcePos_y[:2]
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])

        init = np.array([0, 0.])
        x_true = np.array([0.1, -0.1])
        theta_E = kwargs_lens[0]['theta_E']
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_spep.alpha(x_pos, y_pos, kwargs_lens)
        print(x_pos, 'x_pos')
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        print(a, 'a')
        print(self.solver.F(x_true, x_pos, y_pos, a, theta_E, gamma, e1, e2), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma, 'theta_E': theta_E, 'e1': e1, 'e2': e2})
        x_ = self.solver.F(x, x_pos, y_pos, a, theta_E, gamma, e1, e2)

        [center_x, center_y] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        kwargs_lens_new = [{'theta_E': theta_E, 'gamma': gamma, 'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y}]
        sourcePos_x_new, sourcePos_y_new = self.lens_spep.ray_shooting(x_pos[0], y_pos[0], kwargs_lens_new)
        x_pos_new, y_pos_new = self.image_position_spep.image_position(sourcePos_x_new, sourcePos_y_new, deltapix, numPix, kwargs_lens_new)
        x_pos_new = x_pos_new[:2]
        y_pos_new = y_pos_new[:2]
        print(x_pos_new, 'x_pos_new')
        print(x_pos, 'x_pos old')
        print(kwargs_lens_new)
        npt.assert_almost_equal(x[0], kwargs_lens[0]['center_x'], decimal=2)
        npt.assert_almost_equal(x[1], kwargs_lens[0]['center_y'], decimal=2)

        npt.assert_almost_equal(x_[0], 0, decimal=2)
        npt.assert_almost_equal(x_[1], 0, decimal=2)

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
        x_pos, y_pos = self.image_position_nfw.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        sourcePos_x_new, sourcePos_y_new = self.lens_nfw.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x - sourcePos_x_new, 'sourcePos_x- sourcePos_x_new NFW')
        print(sourcePos_y - sourcePos_y_new, 'sourcePos_y- sourcePos_y_new NFW')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        init = np.array([0, 0.])
        x_true = np.array([0.1, -0.1])
        # [Rs, rho0, r200, center_x_nfw, center_y_nfw] = param
        theta_E = kwargs_lens[0]['theta_E']
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_nfw.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)

        print(self.solver.F(x_true, x_pos, y_pos, a, theta_E, gamma, e1, e2), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma, 'theta_E': theta_E, 'e1': e1, 'e2': e2})
        x_ = self.solver.F(x, x_pos, y_pos, a, theta_E, gamma, e1, e2)
        print(x, 'theta_E, e1, e2, center_x, center_y, non_sens')

        [center_x, center_y] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        Rs = 0.1
        theta_Rs = self.nfw._rho02alpha(1., Rs)
        kwargs_lens_new = [{'theta_E': theta_E, 'gamma': gamma,'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y},
                           {'Rs': Rs, 'theta_Rs': theta_Rs, 'center_x': -0.5, 'center_y': 0.5}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_nfw.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.image_position_nfw.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, deltapix, numPix)
        # plt.plot(x_pos, y_pos, 'or')
        # plt.plot(x_pos_new, y_pos_new, 'og')
        # plt.show()
        print(x_pos_new[:2]-x_pos, 'x_pos_new - x_pos')
        npt.assert_almost_equal(x[0], kwargs_lens[0]['center_x'], decimal=2)
        npt.assert_almost_equal(x[1], kwargs_lens[0]['center_y'], decimal=2)

        npt.assert_almost_equal(x_[0], 0, decimal=2)
        npt.assert_almost_equal(x_[1], 0, decimal=2)

    def test_all_spp(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        kwargs_lens = [{'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = self.image_position_spp.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix, magThresh=1., numImage=4)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        x_mapped, y_mapped = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        print(sourcePos_x - center_x, 'sourcePos_x- sourcePos_x_new SPP')
        print(sourcePos_y - center_y, 'sourcePos_y- sourcePos_y_new SPP')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        init = np.array([0.05, -0.01])
        # init = np.array([1., 1.9, 0.8, 0.5, 0.1, -0.1])
        x_true = np.array([0.1, -0.1])
        theta_E = kwargs_lens[0]['theta_E']
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_spp.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)

        print(self.solver.F(x_true, x_pos, y_pos, a, theta_E, gamma, e1, e2), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'gamma': gamma, 'theta_E': theta_E, 'e1': e1, 'e2': e2})
        x_ = self.solver.F(x, x_pos, y_pos, a, theta_E, gamma, e1, e2)
        print(x, 'center_x, center_y')

        [center_x, center_y] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        kwargs_lens_new = [{'theta_E': theta_E, 'gamma': gamma,'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y},
                           {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.image_position_spp.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, deltapix, numPix)
        print(x_pos_new[:2]-x_pos, 'x_pos_new - x_pos')
        npt.assert_almost_equal(x[0], kwargs_lens[0]['center_x'], decimal=2)
        npt.assert_almost_equal(x[1], kwargs_lens[0]['center_y'], decimal=2)

        npt.assert_almost_equal(x_[0], 0, decimal=2)
        npt.assert_almost_equal(x_[1], 0, decimal=2)


class TestSolverNew(object):

    def setup(self):
        kwargs_options_spep = {'lens_model_list': ['SPEP']}
        self.lens_spep = LensModel(kwargs_options_spep)
        kwargs_options_spep_spp = {'lens_model_list': ['SPEP', 'SPP']}
        self.lens_spep_spp = LensModel(kwargs_options_spep_spp)
        self.image_position_spep_spp = ImagePosition(self.lens_spep_spp)
        kwargs_options_spep_spp_shapelets = {'lens_model_list': ['SHAPELETS_CART', 'SPEP', 'SPP', ]}
        self.lens_spep_spp_shapelets = LensModel(kwargs_options_spep_spp_shapelets)
        self.image_position_spep_spp_shapelets = ImagePosition(self.lens_spep_spp_shapelets)
        self.solverShapelets = SolverShapelets2()
        self.solver = SolverCenter2()
        self.constraints = Constraints2(solver_type='SHAPELETS', lens_model='SHAPELETS_CART')

    def test_all_spp(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        beta = 1.5
        kwargs_lens = [{'coeffs': [0.,-0.1, 0.01], 'beta': beta, 'center_x': 0, 'center_y': 0},
                       {'theta_E': 1., 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                       {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}
                       ]
        x_pos, y_pos = self.image_position_spep_spp_shapelets.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix, magThresh=1., numImage=4)
        x_mapped, y_mapped = self.lens_spep_spp_shapelets.ray_shooting(x_pos, y_pos, kwargs_lens)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        init = np.array([0, 0])
        x_true = np.array(kwargs_lens[0]['coeffs'])[1:]
        # [phi_E, gamma, q, phi_G, center_x, center_y, phi_E_spp, gamma_spp, center_x_spp, center_y_spp] = spep_spp_param
        kwargs_lens[0]['coeffs'] = [0, 0, 0]
        x_sub, y_sub = self.lens_spep_spp_shapelets.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        kwargs = {'beta': kwargs_lens[0]['beta'], 'center_x': kwargs_lens[0]['center_x'], 'center_y': kwargs_lens[0]['center_y']}
        print(self.solverShapelets.F(x_true, x_pos, y_pos, a, **kwargs), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, kwargs)
        x_ = self.solverShapelets.F(x, x_pos, y_pos, a, **kwargs)
        print(x, 'coeffs')

        #[phi_E, q, phi_G, center_x, center_y, no_sens_param] = x
        kwargs_lens_new = [{'coeffs': x, 'beta': beta, 'center_x': 0, 'center_y': 0},
                           {'theta_E': 1., 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1},
                           {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}
                           ]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_spep_spp_shapelets.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        x_pos_new, y_pos_new = self.image_position_spep_spp_shapelets.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, deltapix, numPix)
        print(x_pos_new[:2]-x_pos, 'x_pos_new - x_pos')
        #npt.assert_almost_equal(x[0], kwargs_lens['coeffs'][0], decimal=3)
        npt.assert_almost_equal(x[0], -0.1, decimal=3)
        npt.assert_almost_equal(x[1], 0.01, decimal=3)
        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)


class TestSolver_new(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.constraints = Constraints2_new('SPEP')
        kwargs_options_spep = {'lens_model_list': ['SPEP'], 'fix_center': True}
        self.lens_spep = LensModel(kwargs_options_spep)
        self.image_position_spep = ImagePosition(self.lens_spep)
        kwargs_options_nfw = {'lens_model_list': ['SPEP', 'NFW'], 'fix_center': True}
        self.lens_nfw = LensModel(kwargs_options_nfw)
        self.image_position_nfw = ImagePosition(self.lens_nfw)
        kwargs_options_spp = {'lens_model_list': ['SPEP', 'SPP'], 'fix_center': True}
        self.lens_spp = LensModel(kwargs_options_spp)
        self.image_position_spp = ImagePosition(self.lens_spp)
        self.solver = SolverSPEP2_ellipse()

    def test_subtract(self):
        x_cat = np.array([0, 0])
        y_cat = np.array([1, 2])
        x_sub = np.array([0, 2])
        y_sub = np.array([-1, 0])
        a = self.constraints._subtract_constraint(x_cat, y_cat, x_sub, y_sub)
        assert a[0] == 2

    def test_all_spep(self):
        sourcePos_x = 0.1
        sourcePos_y = 0.03
        deltapix = 0.05
        numPix = 100
        gamma = 1.9
        center_x = 0.1
        center_y = -0.1
        kwargs_lens = [{'theta_E': 1, 'gamma': gamma, 'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1}]
        x_pos, y_pos = self.image_position_spep.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        sourcePos_x, sourcePos_y = self.lens_spep.ray_shooting(x_pos, y_pos, kwargs_lens)
        print(sourcePos_x, sourcePos_y, 'source positions')
        sourcePos_x = sourcePos_x[:2]
        sourcePos_y = sourcePos_y[:2]
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        theta_E = kwargs_lens[0]['theta_E']
        init = np.array([0.9, 2.0]) # theta_E, gamma
        x_true = np.array([theta_E, gamma])

        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_spep.alpha(x_pos, y_pos, kwargs_lens)
        print(x_pos, 'x_pos')
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)
        print(a, 'a')
        print(self.solver.F(x_true, x_pos, y_pos, a, center_x, center_y, e1, e2), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2})
        x_ = self.solver.F(x, x_pos, y_pos, a, center_x, center_y, e1, e2)

        [theta_E_new, gamma_new] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        kwargs_lens_new = [{'theta_E': theta_E_new, 'gamma': gamma_new, 'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y}]
        sourcePos_x_new, sourcePos_y_new = self.lens_spep.ray_shooting(x_pos[0], y_pos[0], kwargs_lens_new)
        x_pos_new, y_pos_new = self.image_position_spep.image_position(sourcePos_x_new, sourcePos_y_new, deltapix, numPix, kwargs_lens_new)
        x_pos_new = x_pos_new[:2]
        y_pos_new = y_pos_new[:2]
        print(x_pos_new, 'x_pos_new')
        print(x_pos, 'x_pos old')
        print(kwargs_lens_new)
        npt.assert_almost_equal(x[0], theta_E, decimal=3)
        npt.assert_almost_equal(x[1], gamma, decimal=3)

        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)

    def test_all_spp(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = 0.05
        numPix = 150
        gamma = 1.9
        theta_E = 1.
        center_x = 0.1
        center_y = -0.1
        kwargs_lens = [{'theta_E': theta_E, 'gamma': gamma,'q': 0.8, 'phi_G': 0.5, 'center_x': center_x, 'center_y': center_y},
                       {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        x_pos, y_pos = self.image_position_spp.findBrightImage(sourcePos_x, sourcePos_y, kwargs_lens, deltapix, numPix, magThresh=1., numImage=4)
        x_pos = x_pos[:2]
        y_pos = y_pos[:2]
        x_mapped, y_mapped = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        print(sourcePos_x - center_x, 'sourcePos_x- sourcePos_x_new SPP')
        print(sourcePos_y - center_y, 'sourcePos_y- sourcePos_y_new SPP')
        e1, e2 = util.phi_q2_elliptisity(kwargs_lens[0]['phi_G'], kwargs_lens[0]['q'])
        init = np.array([0.8, 2.0])
        # init = np.array([1., 1.9, 0.8, 0.5, 0.1, -0.1])
        x_true = np.array([theta_E, gamma])
        theta_E = kwargs_lens[0]['theta_E']
        kwargs_lens[0]['theta_E'] = 0
        x_sub, y_sub = self.lens_spp.alpha(x_pos, y_pos, kwargs_lens)
        a = self.constraints._subtract_constraint(x_pos, y_pos, x_sub, y_sub)

        print(self.solver.F(x_true, x_pos, y_pos, a, center_x, center_y, e1, e2), 'delta true result')
        x = self.constraints.get_param(x_pos, y_pos, x_sub, y_sub, init, {'center_x': center_x, 'center_y': center_y, 'e1': e1, 'e2': e2})
        x_ = self.solver.F(x, x_pos, y_pos, a, center_x, center_y, e1, e2)
        print(x, 'theta_E, gamma')

        [theta_E_new, gamma_new] = x
        phi_G, q = util.elliptisity2phi_q(e1, e2)
        kwargs_lens_new = [{'theta_E': theta_E_new, 'gamma': gamma_new,'q': q, 'phi_G': phi_G, 'center_x': center_x, 'center_y': center_y},
                           {'theta_E': 0.1, 'gamma': 1.9, 'center_x': -0.5, 'center_y': 0.5}]
        sourcePos_x_new_array, sourcePos_y_new_array = self.lens_spp.ray_shooting(x_pos, y_pos, kwargs_lens_new)
        sourcePos_x_new = np.mean(sourcePos_x_new_array)
        sourcePos_y_new = np.mean(sourcePos_y_new_array)
        print(sourcePos_x_new, sourcePos_y_new, 'sourcePos_x_new, sourcePos_y_new')
        print(sourcePos_x_new_array, sourcePos_y_new_array, 'sourcePos_x_new_array, sourcePos_y_new_array')
        x_pos_new, y_pos_new = self.image_position_spp.findBrightImage(sourcePos_x_new, sourcePos_y_new, kwargs_lens_new, deltapix, numPix)
        print(x_pos_new[:2]-x_pos, 'x_pos_new - x_pos')
        npt.assert_almost_equal(x[0], theta_E, decimal=3)
        npt.assert_almost_equal(x[1], gamma, decimal=3)

        npt.assert_almost_equal(x_[0], 0, decimal=3)
        npt.assert_almost_equal(x_[1], 0, decimal=3)


if __name__ == '__main__':
    pytest.main()
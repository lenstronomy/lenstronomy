__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest
from lenstronomy.LensModel.Solver.epl_shear_solver import caustics_epl_shear
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
                                                                     verbose=True, magnification_limit=1)
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
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0}
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
        gamma1, gamma2 = -0.04, -0.1
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}  # shear values to the source plane
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

        x_pos_stoch, y_pos_stoch = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                                solver='stochastic',
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

    def test_analytical_lens_equation_solver(self):
        lensModel = LensModel(['EPL_NUMBA', 'SHEAR'])
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.03
        sourcePos_y = 0.0
        kwargs_lens = [{'theta_E': 1., 'gamma': 2.2, 'center_x': 0.01, 'center_y': 0.02, 'e1': 0.01, 'e2': 0.05},
                        {'gamma1': -0.04, 'gamma2': -0.1, 'ra_0': 0.01, 'dec_0': 0.02}]        

        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, solver='analytical')
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        assert len(source_x) == len(source_y) >= 4
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)
        npt.assert_almost_equal(sourcePos_y, source_y, decimal=10)

        x_pos_ls, y_pos_ls = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens, solver='analytical')
        for x, y in zip(x_pos_ls, y_pos_ls):  # Check if it found all solutions lenstronomy found
            assert np.sqrt((x-x_pos)**2+(y-y_pos)**2).min() < 1e-8

        # here we test with shear and mass profile centroids not aligned
        lensModel = LensModel(['EPL_NUMBA', 'SHEAR'])
        lensEquationSolver = LensEquationSolver(lensModel)
        sourcePos_x = 0.03
        sourcePos_y = 0.0
        kwargs_lens = [{'theta_E': 1., 'gamma': 2.2, 'center_x': 0.01, 'center_y': 0.02, 'e1': 0.01, 'e2': 0.05},
                       {'gamma1': -0.04, 'gamma2': -0.1, 'ra_0': 1.0, 'dec_0': 1.0}]

        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     solver='analytical')
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        assert len(source_x) == len(source_y) >= 2
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)
        npt.assert_almost_equal(sourcePos_y, source_y, decimal=10)

    def test_caustics(self):
        lm = LensModel(['EPL_NUMBA', 'SHEAR'])
        leqs = LensEquationSolver(lm)

        kwargs = [{'theta_E': 1., 'e1': 0.5, 'e2': 0.1, 'center_x': 0.0, 'center_y': 0.0, 'gamma': 1.9},
                  {'gamma1': 0.03, 'gamma2': 0.01, 'ra_0': 0.0, 'dec_0': 0.0}]

        # Calculate the caustics and a few critical curves.
        caus = caustics_epl_shear(kwargs, return_which='caustic')
        lensplane_caus = caustics_epl_shear(kwargs, return_which='caustic', sourceplane=False)
        cut = caustics_epl_shear(kwargs, return_which='cut')
        lensplane_cut = caustics_epl_shear(kwargs, return_which='cut', sourceplane=False)
        twoimg = caustics_epl_shear(kwargs, return_which='double')
        fourimg = caustics_epl_shear(kwargs, return_which='quad')
        assert np.abs(lm.magnification(*lensplane_caus, kwargs)).min() > 1e12
        assert np.abs(lm.magnification(*lensplane_cut, kwargs)).min() > 1e12

        # Test whether the caustics indeed the number of images they say
        N = 20
        xpl, ypl = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
        xgr, ygr = np.meshgrid(xpl, ypl, indexing='ij')
        xf, yf = xgr.flatten(), ygr.flatten()
        sols = [leqs.image_position_from_source(x, y , kwargs, solver='analytical')
                for x, y in zip(xf, yf)]
        numsols = np.array([len(p[0]) for p in sols])

        from matplotlib.path import Path

        points = np.vstack((xf, yf)).T

        p = Path(twoimg.T)  # make a polygon
        grid2img = p.contains_points(points)
        assert np.all(numsols[grid2img] >= 2)

        p = Path(fourimg.T)  # make a polygon
        grid4img = p.contains_points(points)
        assert np.all(numsols[grid4img] >= 4)

    def test_analytical_sie(self):
        sourcePos_x = 0.03
        sourcePos_y = 0.0

        lensModel = LensModel(['SIE'])
        lensEquationSolver = LensEquationSolver(lensModel)
        kwargs_lens = [{'theta_E': 1., 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.5, 'e2': 0.05}, ]

        x_pos, y_pos = lensEquationSolver.image_position_from_source(sourcePos_x, sourcePos_y, kwargs_lens,
                                                                     solver='analytical', magnification_limit=1e-3)
        source_x, source_y = lensModel.ray_shooting(x_pos, y_pos, kwargs_lens)
        assert len(source_x) == len(source_y) == 4
        npt.assert_almost_equal(sourcePos_x, source_x, decimal=10)
        npt.assert_almost_equal(sourcePos_y, source_y, decimal=10)

    def test_assertions(self):
        lensModel = LensModel(['SPEP'])
        lensEquationSolver = LensEquationSolver(lensModel)
        kwargs_lens = [{'theta_E': 1, 'gamma': 2, 'e1': 0.2, 'e2': -0.03, 'center_x': 0, 'center_y': 0}]
        with pytest.raises(ValueError):
            lensEquationSolver.image_position_from_source(0.1, 0., kwargs_lens, solver='analytical')

        lensModel = LensModel(['EPL_NUMBA', 'SHEAR'])
        lensEquationSolver = LensEquationSolver(lensModel)
        kwargs_lens = [{'theta_E': 1., 'gamma': 2.2, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.01, 'e2': 0.05},
                       {'gamma1': -0.04, 'gamma2': -0.1, 'ra_0': 0.0, 'dec_0': 0.0}]

        with pytest.raises(ValueError):
            lensEquationSolver.image_position_from_source(0.1, 0., kwargs_lens, solver='nonexisting')

        # with pytest.raises(ValueError):
        #    kwargs_lens[1]['ra_0']=0.1
        #    lensEquationSolver.image_position_from_source(0.1, 0., kwargs_lens, solver='analytical')



if __name__ == '__main__':
    pytest.main()

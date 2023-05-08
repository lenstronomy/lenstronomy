import pytest
import numpy as np
import numpy.testing as npt
import unittest

from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.param_util as param_util


class TestPointSource(object):

    def setup_method(self):
        lensModel = LensModel(lens_model_list=['SPEP'])
        solver = LensEquationSolver(lensModel=lensModel)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        self.kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2, 'gamma': 2}]
        self.sourcePos_x, self.sourcePos_y = 0.01, -0.01
        self.x_pos, self.y_pos = solver.image_position_from_source(sourcePos_x=self.sourcePos_x,
                                                                   sourcePos_y=self.sourcePos_y, kwargs_lens=self.kwargs_lens)
        self.PointSource = PointSource(point_source_type_list=['LENSED_POSITION', 'UNLENSED', 'SOURCE_POSITION'],
                                       lensModel=lensModel, fixed_magnification_list=[False]*3,
                                       additional_images_list=[False]*4, flux_from_point_source_list=[True, True, True])
        self.kwargs_ps = [{'ra_image': self.x_pos, 'dec_image': self.y_pos, 'point_amp': np.ones_like(self.x_pos) * 2},
                          {'ra_image': [1.], 'dec_image': [1.], 'point_amp': [10]},
                          {'ra_source': self.sourcePos_x, 'dec_source': self.sourcePos_y, 'point_amp': np.ones_like(self.x_pos)}, {}]

    def test_image_position(self):
        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=self.kwargs_ps, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(x_image_list[0][0], self.x_pos[0], decimal=8)
        npt.assert_almost_equal(x_image_list[1], 1, decimal=8)
        npt.assert_almost_equal(x_image_list[2][0], self.x_pos[0], decimal=8)

        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=self.kwargs_ps,
                                                                     kwargs_lens=self.kwargs_lens,
                                                                     original_position=True, additional_images=True)
        npt.assert_almost_equal(x_image_list[0][0], self.x_pos[0], decimal=8)
        npt.assert_almost_equal(x_image_list[1], 1, decimal=8)
        npt.assert_almost_equal(x_image_list[2][0], self.x_pos[0], decimal=8)

    def test_source_position(self):
        x_source_list, y_source_list = self.PointSource.source_position(kwargs_ps=self.kwargs_ps, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(x_source_list[0], self.sourcePos_x, decimal=8)
        npt.assert_almost_equal(x_source_list[1], 1, decimal=8)
        npt.assert_almost_equal(x_source_list[2], self.sourcePos_x, decimal=8)

    def test_num_basis(self):
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert num_basis == 9

    def test_linear_response_set(self):
        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens, with_amp=False)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert amp[0][0] == 1
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]

        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens,
                                                                       with_amp=True)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert amp[0][0] != 1
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]

    def test_linear_param_from_kwargs(self):
        param = self.PointSource.linear_param_from_kwargs(self.kwargs_ps)
        assert param[0] == self.kwargs_ps[0]['point_amp'][0]
        assert param[1] == self.kwargs_ps[0]['point_amp'][1]

    def test_point_source_list(self):
        ra_list, dec_list, amp_list = self.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens)
        assert ra_list[0] == self.x_pos[0]
        assert len(ra_list) == 9

        ra_list, dec_list, amp_list = self.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens, k=0)
        assert ra_list[0] == self.x_pos[0]
        assert len(ra_list) == 4
        assert len(dec_list) == 4
        assert len(amp_list) == 4

    def test_point_source_amplitude(self):
        amp_list = self.PointSource.source_amplitude(self.kwargs_ps, self.kwargs_lens)
        assert len(amp_list) == 3

    def test_set_save_cache(self):
        self.PointSource.set_save_cache(True)
        assert self.PointSource._point_source_list[0]._save_cache == True

        self.PointSource.set_save_cache(False)
        assert self.PointSource._point_source_list[0]._save_cache == False

    def test_update_lens_model(self):
        lensModel = LensModel(lens_model_list=['SIS'])
        self.PointSource.update_lens_model(lens_model_class=lensModel)
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=self.kwargs_ps,
                                                                     kwargs_lens=kwargs_lens)
        npt.assert_almost_equal(x_image_list[0][-1], -0.82654997748011705 , decimal=8)

    def test_set_amplitudes(self):
        amp_list = [np.ones_like(self.x_pos)*20, [100], np.ones_like(self.x_pos)*10]
        kwargs_out = self.PointSource.set_amplitudes(amp_list, self.kwargs_ps)
        assert kwargs_out[0]['point_amp'][0] == 10 * self.kwargs_ps[0]['point_amp'][0]
        assert kwargs_out[1]['point_amp'][0] == 10 * self.kwargs_ps[1]['point_amp'][0]
        assert kwargs_out[2]['point_amp'][3] == 10 * self.kwargs_ps[2]['point_amp'][3]

    def test_update_search_window(self):
        search_window = 5
        x_center, y_center = 1, 1
        min_distance = 0.01

        point_source = PointSource(point_source_type_list=['LENSED_POSITION'],
                                   lensModel=None, kwargs_lens_eqn_solver={})

        point_source.update_search_window(search_window, x_center, y_center, min_distance=min_distance, only_from_unspecified=False)
        assert point_source._kwargs_lens_eqn_solver['search_window'] == search_window
        assert point_source._kwargs_lens_eqn_solver['x_center'] == x_center
        assert point_source._kwargs_lens_eqn_solver['x_center'] == y_center

        point_source = PointSource(point_source_type_list=['LENSED_POSITION'],
                                   lensModel=None, kwargs_lens_eqn_solver={})

        point_source.update_search_window(search_window, x_center, y_center, min_distance=min_distance,
                                          only_from_unspecified=True)
        assert point_source._kwargs_lens_eqn_solver['search_window'] == search_window
        assert point_source._kwargs_lens_eqn_solver['x_center'] == x_center
        assert point_source._kwargs_lens_eqn_solver['x_center'] == y_center

        kwargs_lens_eqn_solver = {'search_window': search_window,
                                  'min_distance': min_distance, 'x_center': x_center, 'y_center': y_center}
        point_source = PointSource(point_source_type_list=['LENSED_POSITION'],
                                   lensModel=None, kwargs_lens_eqn_solver=kwargs_lens_eqn_solver)
        point_source.update_search_window(search_window=-10, x_center=-10, y_center=-10,
                                          min_distance=10, only_from_unspecified = True)
        assert point_source._kwargs_lens_eqn_solver['search_window'] == search_window
        assert point_source._kwargs_lens_eqn_solver['x_center'] == x_center
        assert point_source._kwargs_lens_eqn_solver['x_center'] == y_center

    def test__sort_position_by_original(self):
        from lenstronomy.PointSource.point_source import _sort_position_by_original
        x_o, y_o = np.array([1, 2]), np.array([0, 0])
        x_solved, y_solved = np.array([2]), np.array([0])
        x_new, y_new = _sort_position_by_original(x_o, y_o, x_solved, y_solved)
        npt.assert_almost_equal(x_new, x_o, decimal=7)
        npt.assert_almost_equal(y_new, y_o, decimal=7)

        x_solved, y_solved = np.array([2, 1]), np.array([0, 0.01])
        x_new, y_new = _sort_position_by_original(x_o, y_o, x_solved, y_solved)
        npt.assert_almost_equal(x_new, x_o, decimal=7)
        npt.assert_almost_equal(y_new, np.array([0.01, 0]), decimal=7)


class TestPointSourceFixedMag(object):

    def setup_method(self):
        lensModel = LensModel(lens_model_list=['SPEP'])
        solver = LensEquationSolver(lensModel=lensModel)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        self.kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2, 'gamma': 2}]
        self.sourcePos_x, self.sourcePos_y = 0.01, -0.01
        self.x_pos, self.y_pos = solver.image_position_from_source(sourcePos_x=self.sourcePos_x,
                                                                   sourcePos_y=self.sourcePos_y, kwargs_lens=self.kwargs_lens)
        self.PointSource = PointSource(point_source_type_list=['LENSED_POSITION', 'UNLENSED', 'SOURCE_POSITION'],
                                       lensModel=lensModel, fixed_magnification_list=[True]*4, additional_images_list=[False]*4)
        self.kwargs_ps = [{'ra_image': self.x_pos, 'dec_image': self.y_pos, 'source_amp': 1},
                          {'ra_image': [1.], 'dec_image': [1.], 'point_amp': [10]},
                          {'ra_source': self.sourcePos_x, 'dec_source': self.sourcePos_y, 'source_amp': 1.}, {}]

    def test_image_position(self):
        x_image_list, y_image_list = self.PointSource.image_position(kwargs_ps=self.kwargs_ps, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(x_image_list[0][0], self.x_pos[0], decimal=8)
        npt.assert_almost_equal(x_image_list[1], 1, decimal=8)
        npt.assert_almost_equal(x_image_list[2][0], self.x_pos[0], decimal=8)

    def test_source_position(self):
        x_source_list, y_source_list = self.PointSource.source_position(kwargs_ps=self.kwargs_ps, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(x_source_list[0], self.sourcePos_x, decimal=8)
        npt.assert_almost_equal(x_source_list[1], 1, decimal=8)
        npt.assert_almost_equal(x_source_list[2], self.sourcePos_x, decimal=8)

    def test_num_basis(self):
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert num_basis == 3

    def test_linear_response_set(self):
        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens,
                                                                       with_amp=False)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]
        assert ra_pos[1][0] == 1
        assert np.all(amp != 1)
        npt.assert_almost_equal(ra_pos[2][0], self.x_pos[0], decimal=8)

        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens,
                                                                       with_amp=True)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]
        assert ra_pos[1][0] == 1
        assert np.all(amp != 1)
        npt.assert_almost_equal(ra_pos[2][0], self.x_pos[0], decimal=8)

    def test_point_source_list(self):
        ra_list, dec_list, amp_list = self.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens)
        assert ra_list[0] == self.x_pos[0]
        assert len(ra_list) == 9

    def test_check_image_positions(self):
        bool = self.PointSource.check_image_positions(self.kwargs_ps, self.kwargs_lens, tolerance=0.001)
        assert bool is True

        # now we change the lens model to make the test fail
        kwargs_lens = [{'theta_E': 2., 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0, 'gamma': 2}]
        bool = self.PointSource.check_image_positions(self.kwargs_ps, kwargs_lens, tolerance=0.001)
        assert bool is False

    def test_set_amplitudes(self):
        amp_list = [10, [100], 10]
        kwargs_out = self.PointSource.set_amplitudes(amp_list, self.kwargs_ps)
        assert kwargs_out[0]['source_amp'] == 10 * self.kwargs_ps[0]['source_amp']
        assert kwargs_out[1]['point_amp'][0] == 10 * self.kwargs_ps[1]['point_amp'][0]
        assert kwargs_out[2]['source_amp'] == 10 * self.kwargs_ps[2]['source_amp']

    def test_positive_flux(self):
        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': np.array([1, -1])}])
        assert bool is False
        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': -1}])
        assert bool is False

        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': np.array([0, 1])}])
        assert bool is True
        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': 1}])
        assert bool is True

        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': np.array([0, 1]), 'source_amp': 1}])
        assert bool is True
        bool = PointSource.check_positive_flux(kwargs_ps=[{'point_amp': 1, 'source_amp': -1}])
        assert bool is False


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            PointSource(point_source_type_list=['BAD'])


if __name__ == '__main__':
    pytest.main()

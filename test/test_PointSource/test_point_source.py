import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.param_util as param_util


class TestPointSource(object):

    def setup(self):
        lensModel = LensModel(lens_model_list=['SPEP'])
        solver = LensEquationSolver(lensModel=lensModel)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.7)
        self.kwargs_lens = [{'theta_E': 1., 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2, 'gamma': 2}]
        self.sourcePos_x, self.sourcePos_y = 0.01, -0.01
        self.x_pos, self.y_pos = solver.image_position_from_source(sourcePos_x=self.sourcePos_x,
                                                                   sourcePos_y=self.sourcePos_y, kwargs_lens=self.kwargs_lens)
        self.PointSource = PointSource(point_source_type_list=['LENSED_POSITION', 'UNLENSED', 'SOURCE_POSITION'],
                                       lensModel=lensModel, fixed_magnification_list=[False]*4, additional_images_list=[False]*4)
        self.kwargs_ps = [{'ra_image': self.x_pos, 'dec_image': self.y_pos, 'point_amp': np.ones_like(self.x_pos)},
                          {'ra_image': [1.], 'dec_image': [1.], 'point_amp': [10]},
                          {'ra_source': self.sourcePos_x, 'dec_source': self.sourcePos_y, 'point_amp': np.ones_like(self.x_pos)}, {}]

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
        assert num_basis == 9

    def test_linear_response_set(self):
        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens, with_amp=False, k=None)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]

    def test_point_source_list(self):
        ra_list, dec_list, amp_list = self.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens)
        assert ra_list[0] == self.x_pos[0]
        assert len(ra_list) == 9

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
        npt.assert_almost_equal(x_image_list[0][0], -0.82654997748011705 , decimal=8)


class TestPointSource_fixed_mag(object):

    def setup(self):
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
        ra_pos, dec_pos, amp, n = self.PointSource.linear_response_set(self.kwargs_ps, kwargs_lens=self.kwargs_lens, with_amp=False, k=None)
        num_basis = self.PointSource.num_basis(self.kwargs_ps, self.kwargs_lens)
        assert n == num_basis
        assert ra_pos[0][0] == self.x_pos[0]
        assert ra_pos[1][0] == 1
        npt.assert_almost_equal(ra_pos[2][0], self.x_pos[0], decimal=8)

    def test_point_source_list(self):
        ra_list, dec_list, amp_list = self.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens)
        assert ra_list[0] == self.x_pos[0]
        assert len(ra_list) == 9

    def test_check_image_positions(self):
        bool = self.PointSource.check_image_positions(self.kwargs_ps, self.kwargs_lens, tolerance=0.001)
        assert bool == True


if __name__ == '__main__':
    pytest.main()

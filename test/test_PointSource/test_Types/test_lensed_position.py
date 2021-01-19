from lenstronomy.PointSource.Types.lensed_position import LensedPositions
from lenstronomy.LensModel.lens_model import LensModel
import pytest
import numpy.testing as npt


class TestLensedPosition(object):

    def setup(self):
        lens_model = LensModel(lens_model_list=['SIS'])
        self.kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        self.ps_mag = LensedPositions(lens_model=lens_model, fixed_magnification=True)
        self.ps = LensedPositions(lens_model=lens_model, fixed_magnification=False)
        self.ps_add = LensedPositions(lens_model=lens_model, fixed_magnification=[False], additional_image=True)
        self.kwargs = {'point_amp': [2, 1], 'ra_image': [0, 1.2], 'dec_image': [0, 0]}
        self.kwargs_mag = {'source_amp': 2, 'ra_image': [0, 1.2], 'dec_image': [0, 0]}

    def test_image_source_position(self):
        x_img, y_img = self.ps.image_position(self.kwargs, self.kwargs_lens)
        npt.assert_almost_equal(x_img, self.kwargs['ra_image'])

        x_img_add, y_img_add = self.ps_add.image_position(self.kwargs, self.kwargs_lens)
        print(x_img_add, x_img)
        assert x_img[0] != x_img_add[0]

        # check whether the source solution matches
        x_src, y_src = self.ps.source_position(self.kwargs, self.kwargs_lens)
        lens_model = LensModel(lens_model_list=['SIS'])
        x_src_true, y_src_true = lens_model.ray_shooting(x_img_add, y_img_add, kwargs=self.kwargs_lens)
        npt.assert_almost_equal(x_src_true[0], x_src_true[1])
        npt.assert_almost_equal(y_src_true[0], y_src_true[1])

        npt.assert_almost_equal(x_src_true, x_src)
        npt.assert_almost_equal(y_src_true, y_src)

    def test_image_amplitude(self):
        amp = self.ps.image_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens, x_pos=self.kwargs['ra_image'],
                                      y_pos=self.kwargs['dec_image'], magnification_limit=None,
                                      kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(self.kwargs['point_amp'], amp)

        amp = self.ps_mag.image_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens, x_pos=None,
                                          y_pos=None, magnification_limit=None, kwargs_lens_eqn_solver=None)

        amp_pos = self.ps_mag.image_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens, x_pos=self.kwargs['ra_image'],
                                              y_pos=self.kwargs['dec_image'], magnification_limit=None,
                                              kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(amp, amp_pos)

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens)
        amp_mag = self.ps_mag.source_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(amp_mag, self.kwargs_mag['source_amp'])
        assert amp != amp_mag


if __name__ == '__main__':
    pytest.main()

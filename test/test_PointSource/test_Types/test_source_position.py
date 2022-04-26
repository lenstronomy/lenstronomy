from lenstronomy.PointSource.Types.source_position import SourcePositions
from lenstronomy.LensModel.lens_model import LensModel
import pytest
import numpy.testing as npt


class TestLensedPosition(object):

    def setup(self):
        lens_model = LensModel(lens_model_list=['SIS'])
        self.kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        self.ps_mag = SourcePositions(lens_model=lens_model, fixed_magnification=True)
        self.ps = SourcePositions(lens_model=lens_model, fixed_magnification=False)
        self.kwargs = {'point_amp': [2, 1], 'ra_source': 0.1, 'dec_source': 0}
        self.kwargs_mag = {'source_amp': 1, 'ra_source': 0.1, 'dec_source': 0}

    def test_image_position(self):
        x_img, y_img = self.ps.image_position(self.kwargs, self.kwargs_lens)

        lens_model = LensModel(lens_model_list=['SIS'])
        x_src, y_src = lens_model.ray_shooting(x_img, y_img, kwargs=self.kwargs_lens)
        npt.assert_almost_equal(x_src, self.kwargs['ra_source'])
        npt.assert_almost_equal(y_src, self.kwargs['dec_source'])

    def test_source_position(self):
        x_src, y_src = self.ps.source_position(self.kwargs, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(x_src, self.kwargs['ra_source'])
        npt.assert_almost_equal(y_src, self.kwargs['dec_source'])

    def test_image_amplitude(self):
        x_img, y_img = self.ps.image_position(self.kwargs, kwargs_lens=self.kwargs_lens)
        amp = self.ps_mag.image_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens, x_pos=None,
                                          y_pos=None, magnification_limit=None, kwargs_lens_eqn_solver=None)
        amp_pos = self.ps_mag.image_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens, x_pos=x_img,
                                              y_pos=y_img, magnification_limit=None, kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(amp_pos, amp)

        amp = self.ps.image_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens, x_pos=x_img,
                                              y_pos=y_img, magnification_limit=None,
                                              kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(amp, self.kwargs['point_amp'])

        #see if works with mag_pert defined
        self.kwargs['mag_pert'] = [0.1,0.1]
        amp_pert = self.ps.image_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens, x_pos=x_img,
                                              y_pos=y_img, magnification_limit=None,
                                              kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(amp_pert, 0.1*amp)

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(self.kwargs, kwargs_lens=self.kwargs_lens)
        amp_mag = self.ps_mag.source_amplitude(self.kwargs_mag, kwargs_lens=self.kwargs_lens)
        npt.assert_almost_equal(amp_mag, self.kwargs_mag['source_amp'])
        assert amp != amp_mag


if __name__ == '__main__':
    pytest.main()

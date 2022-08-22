from lenstronomy.PointSource.Types.unlensed import Unlensed
import pytest
import numpy.testing as npt


class TestUnlensed(object):

    def setup(self):

        self.ps = Unlensed()
        self.kwargs = {'image_amp': [2, 1], 'ra_image': [0, 1], 'dec_image': [1, 0]}

    def test_image_position(self):
        x_img, y_img = self.ps.image_position(self.kwargs)
        npt.assert_almost_equal(x_img, self.kwargs['ra_image'])
        npt.assert_almost_equal(y_img, self.kwargs['dec_image'])

    def test_source_position(self):
        x_src, y_src = self.ps.source_position(self.kwargs, kwargs_lens=None)
        npt.assert_almost_equal(x_src, self.kwargs['ra_image'])
        npt.assert_almost_equal(y_src, self.kwargs['dec_image'])

    def test_image_amplitude(self):
        amp = self.ps.image_amplitude(self.kwargs, kwargs_lens=None, x_pos=None,
                                      y_pos=None, magnification_limit=None, kwargs_lens_eqn_solver=None)
        npt.assert_almost_equal(amp, self.kwargs['image_amp'])

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(self.kwargs, kwargs_lens=None)
        npt.assert_almost_equal(amp, self.kwargs['image_amp'])


if __name__ == '__main__':
    pytest.main()

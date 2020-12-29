from lenstronomy.PointSource.point_source_cached import PointSourceCached
from lenstronomy.PointSource.Types.unlensed import Unlensed
import numpy.testing as npt
import pytest


class TestPointSourceCached(object):

    def setup(self):
        self.ps_cached = PointSourceCached(Unlensed(), save_cache=True)
        self.ps = Unlensed()
        self.kwargs_ps = {'ra_image': [1], 'dec_image': [0], 'point_amp': [1]}
        self.kwargs_ps_dummy = {'ra_image': [-1], 'dec_image': [10], 'point_amp': [-1]}

    def test_image_position(self):
        x_img, y_img = self.ps.image_position(kwargs_ps=self.kwargs_ps)
        x_img_cached, y_img_cached = self.ps_cached.image_position(kwargs_ps=self.kwargs_ps)
        npt.assert_almost_equal(x_img_cached, x_img)
        npt.assert_almost_equal(y_img_cached, y_img)

        x_img_cached, y_img_cached = self.ps_cached.image_position(kwargs_ps=self.kwargs_ps_dummy)
        npt.assert_almost_equal(x_img_cached, x_img)
        npt.assert_almost_equal(y_img_cached, y_img)

        self.ps_cached.delete_lens_model_cache()
        x_img_cached, y_img_cached = self.ps_cached.image_position(kwargs_ps=self.kwargs_ps_dummy)
        assert x_img_cached[0] != x_img[0]

    def test_source_position(self):
        x_img, y_img = self.ps.source_position(kwargs_ps=self.kwargs_ps)
        x_img_cached, y_img_cached = self.ps_cached.source_position(kwargs_ps=self.kwargs_ps)
        npt.assert_almost_equal(x_img_cached, x_img)
        npt.assert_almost_equal(y_img_cached, y_img)

        x_img_cached, y_img_cached = self.ps_cached.source_position(kwargs_ps=self.kwargs_ps_dummy)
        npt.assert_almost_equal(x_img_cached, x_img)
        npt.assert_almost_equal(y_img_cached, y_img)

        self.ps_cached.delete_lens_model_cache()
        x_img_cached, y_img_cached = self.ps_cached.source_position(kwargs_ps=self.kwargs_ps_dummy)
        assert x_img_cached[0] != x_img[0]

    def test_image_amplitude(self):
        amp = self.ps.image_amplitude(kwargs_ps=self.kwargs_ps)
        amp_cached = self.ps_cached.image_amplitude(kwargs_ps=self.kwargs_ps)
        npt.assert_almost_equal(amp_cached, amp)

        # amplitudes are not cached!!!
        amp_cached = self.ps_cached.image_amplitude(kwargs_ps=self.kwargs_ps_dummy)
        assert amp_cached[0] != amp[0]

        self.ps_cached.delete_lens_model_cache()
        amp_cached = self.ps_cached.image_amplitude(kwargs_ps=self.kwargs_ps_dummy)
        assert amp_cached[0] != amp[0]

    def test_source_amplitude(self):
        amp = self.ps.source_amplitude(kwargs_ps=self.kwargs_ps)
        amp_cached = self.ps_cached.source_amplitude(kwargs_ps=self.kwargs_ps)
        npt.assert_almost_equal(amp_cached, amp)

        # amplitudes are not cached!!!
        amp_cached = self.ps_cached.source_amplitude(kwargs_ps=self.kwargs_ps_dummy)
        assert amp_cached[0] != amp[0]

        self.ps_cached.delete_lens_model_cache()
        amp_cached = self.ps_cached.source_amplitude(kwargs_ps=self.kwargs_ps_dummy)
        assert amp_cached[0] != amp[0]


if __name__ == '__main__':
    pytest.main()

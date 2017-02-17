__author__ = 'sibirrer'


import pytest
from mock import patch
import numpy as np
import numpy.testing as npt
import astrofunc.util as util

from lenstronomy.ImSim.make_image import MakeImage

class TestMakeImage(object):
    """
    tests the source model routines
    """
    @patch("darkskysync.DarkSkySync", autospec=False)
    def setup(self, dss_mock):
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_type': 'GAUSSIAN', 'source_type': 'GAUSSIAN', 'lens_light_type': 'TRIPPLE_SERSIC'
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        self.kwargs_data = {}
        
        self.makeImage = MakeImage(self.kwargs_options, self.kwargs_data)
        self.kwargs = {'amp': 1, 'sigma_x': 2, 'sigma_y': 2,'center_x': 0, 'center_y': 0}
        x_grid, y_grid = util.make_grid(numPix = 100,deltapix = 0.1)
        x_source, y_source = self.makeImage.mapping_IS(x_grid, y_grid, **self.kwargs)
        I_xy = self.makeImage.get_surface_brightness(x_source, y_source, **self.kwargs)
        self.grid = util.array2image(I_xy)
        np.random.seed(seed=41)

    def test_mapping_IS(self):
        delta_x, delta_y = self.makeImage.mapping_IS(x=1., y=1., **self.kwargs)
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

    def test_get_surface_brightness(self):
        I_xy = self.makeImage.get_surface_brightness(x=1., y=1., **self.kwargs)
        assert I_xy == 0.77880078307140488/(8*np.pi)

    def test_psf_convolution(self):
        kwargs = {'sigma': 1}
        grid_convolved = self.makeImage.psf_convolution(self.grid, 1., **kwargs)
        assert (grid_convolved[0][0] > 8.447e-05 and grid_convolved[0][0] < 8.448e-05)

    def test_estimate_amp(self):
        data = np.ones((20, 20))
        psf_kernel = np.ones((20, 20))/10.
        x_pos = 9
        y_pos = 9.5
        mag = self.makeImage.estimate_amp(data, x_pos, y_pos, psf_kernel)
        npt.assert_almost_equal(mag, 10, decimal=10)

        data[5, 5] = 0
        mag = self.makeImage.estimate_amp(data, x_pos, y_pos, psf_kernel)
        npt.assert_almost_equal(mag, 10, decimal=10)

    def test_get_magnification_model(self):
        kwargs_else = {'ra_pos': np.array([1., 1., 2.]), 'dec_pos': np.array([-1., 0., 0.])}
        x_pos, y_pos, mag = self.makeImage.get_magnification_model(self.kwargs, kwargs_else)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_get_image_amplitudes(self):
        param = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        kwargs_else = {'ra_pos': np.array([1, 1, 2]), 'dec_pos': np.array([-1, 0, 0])}
        mag, _ = self.makeImage.get_image_amplitudes(param, kwargs_else)
        assert mag[0] == 5
        assert mag[1] == 6
        assert mag[2] == 7

if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'

import astrofunc.util as util
import numpy as np
import numpy.testing as npt
import pytest
from mock import patch

from lenstronomy.ImSim.make_image import MakeImage


class TestMakeImage(object):
    """
    tests the source model routines
    """
    @patch("darkskysync.DarkSkySync", autospec=False)
    def setup(self, dss_mock):
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_model_list': ['GAUSSIAN'], 'source_light_model_list': ['GAUSSIAN'], 'lens_light_model_list': ['SERSIC']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'gaussian', 'x2_simple': True}
        self.kwargs_data = {}
        
        self.makeImage = MakeImage(self.kwargs_options, self.kwargs_data)
        self.kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2,'center_x': 0, 'center_y': 0}]
        self.kwargs_source = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        x_grid, y_grid = util.make_grid(numPix=101, deltapix=0.1)
        x_source, y_source = self.makeImage.mapping_IS(x_grid, y_grid, self.kwargs_lens)
        I_xy = self.makeImage.get_surface_brightness(x_source, y_source, self.kwargs_source)
        self.grid = util.array2image(I_xy)
        np.random.seed(seed=41)

    def test_mapping_IS(self):
        delta_x, delta_y = self.makeImage.mapping_IS(x=1., y=1., kwargs=self.kwargs_lens)
        assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

    def test_get_surface_brightness(self):
        I_xy = self.makeImage.get_surface_brightness(x=np.array([1.]), y=np.array([1.]), kwargs=self.kwargs_source)
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
        x_pos, y_pos, mag = self.makeImage.get_magnification_model(self.kwargs_lens, kwargs_else)
        npt.assert_almost_equal(mag[0], 0.98848384784633392, decimal=5)

    def test_get_image_amplitudes(self):
        param = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        kwargs_else = {'ra_pos': np.array([1, 1, 2]), 'dec_pos': np.array([-1, 0, 0])}
        mag, _ = self.makeImage.get_image_amplitudes(param, kwargs_else)
        assert mag[0] == 2
        assert mag[1] == 3
        assert mag[2] == 4

    def test_add_mask(self):
        mask = [0, 1, 0]
        A = np.ones((10, 3))
        A_masked = self.makeImage._add_mask(A, mask)
        assert A[0, 1] == A_masked[0, 1]
        assert A_masked[0, 2] == 0

    def test_image2array(self):
        self.makeImage._idex_mask = 0
        image = np.ones((10, 10))
        image_array = util.image2array(image)
        idex_mask = np.zeros_like(image_array)
        #idex_mask[0, 3, 8, 59, 66] = 1
        #array = self.makeImage.image2array(image)

    def test_idex_subgrid(self):
        idex_mask = np.zeros(100)
        n = 8
        nx, ny = np.sqrt(len(idex_mask)), np.sqrt(len(idex_mask))
        idex_mask[n] = 1
        subgrid_res = 2
        idex_mask_subgrid = self.makeImage._subgrid_idex(idex_mask, subgrid_res, nx, ny)
        assert idex_mask_subgrid[(n+1)*subgrid_res-1] == 1
        assert idex_mask_subgrid[(n+1)*subgrid_res-2] == 1
        assert idex_mask_subgrid[nx*subgrid_res + (n+1)*subgrid_res-1] == 1


if __name__ == '__main__':
    pytest.main()
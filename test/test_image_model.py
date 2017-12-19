__author__ = 'sibirrer'

import astrofunc.util as util
import numpy as np
import pytest
from mock import patch

from lenstronomy.ImSim.image_model import ImageModel


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
        
        self.makeImage = ImageModel(self.kwargs_options, self.kwargs_data)
        self.kwargs_lens = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2,'center_x': 0, 'center_y': 0}]
        self.kwargs_source = [{'amp': 1, 'sigma_x': 2, 'sigma_y': 2, 'center_x': 0, 'center_y': 0}]
        x_grid, y_grid = util.make_grid(numPix=101, deltapix=0.1)
        x_source, y_source = self.makeImage.LensModel.ray_shooting(x_grid, y_grid, self.kwargs_lens)
        I_xy = self.makeImage.SourceModel.surface_brightness(x_source, y_source, self.kwargs_source)
        self.grid = util.array2image(I_xy)
        np.random.seed(seed=41)

    def test_psf_convolution(self):
        kwargs = {'sigma': 1, 'psf_type': 'gaussian'}
        grid_convolved = self.makeImage.Data.psf_convolution(self.grid, 1., **kwargs)
        assert (grid_convolved[0][0] > 8.447e-05 and grid_convolved[0][0] < 8.448e-05)

    def test_add_mask(self):
        mask = [0, 1, 0]
        A = np.ones((10, 3))
        A_masked = self.makeImage._add_mask(A, mask)
        assert A[0, 1] == A_masked[0, 1]
        assert A_masked[0, 2] == 0

    def test_point_source_rendering(self):
        # initialize data
        from lenstronomy.Extensions.SimulationAPI.simulations import Simulation
        SimAPI = Simulation()
        numPix = 100
        deltaPix = 0.05
        kwargs_data = SimAPI.data_configure(numPix, deltaPix, exposure_time=1, sigma_bkg=1)
        kwargs_options = {'lens_model_list': ['SPEP'], 'point_source': True, 'subgrid_res': 2}
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {'kernel_point_source': kernel, 'kernel_pixel': kernel, 'psf_type': 'pixel'}
        makeImage = ImageModel(kwargs_options, kwargs_data, kwargs_psf=kwargs_psf)
        # chose point source positions
        x_pix = np.array([10, 5, 10, 90])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'q': 0.8, 'phi_G': 0, 'center_x': 0, 'center_y': 0}]
        kwargs_else = {'ra_pos': ra_pos, 'dec_pos': dec_pos, 'point_amp': np.ones_like(ra_pos)}
        model, _ = makeImage.image_with_params(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_else=kwargs_else)
        image = makeImage.Data.array2image(model)
        for i in range(len(x_pix)):
            assert image[y_pix[i], x_pix[i]] == 1

        x_pix = np.array([10.5, 5.5, 10.5, 90.5])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'q': 0.8, 'phi_G': 0, 'center_x': 0, 'center_y': 0}]
        kwargs_else = {'ra_pos': ra_pos, 'dec_pos': dec_pos, 'point_amp': np.ones_like(ra_pos)}
        model, _ = makeImage.image_with_params(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_else=kwargs_else)
        image = makeImage.Data.array2image(model)
        for i in range(len(x_pix)):
            print(int(y_pix[i]), int(x_pix[i]+0.5))
            assert image[int(y_pix[i]), int(x_pix[i])] == 0.5
            assert image[int(y_pix[i]), int(x_pix[i]+0.5)] == 0.5


if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'

import astrofunc.util as util
import numpy as np
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
        x_source, y_source = self.makeImage.LensModel.ray_shooting(x_grid, y_grid, self.kwargs_lens)
        I_xy = self.makeImage.SourceModel.surface_brightness(x_source, y_source, self.kwargs_source)
        self.grid = util.array2image(I_xy)
        np.random.seed(seed=41)

    def test_psf_convolution(self):
        kwargs = {'sigma': 1}
        grid_convolved = self.makeImage.Data.psf_convolution(self.grid, 1., **kwargs)
        assert (grid_convolved[0][0] > 8.447e-05 and grid_convolved[0][0] < 8.448e-05)

    def test_add_mask(self):
        mask = [0, 1, 0]
        A = np.ones((10, 3))
        A_masked = self.makeImage._add_mask(A, mask)
        assert A[0, 1] == A_masked[0, 1]
        assert A_masked[0, 2] == 0


if __name__ == '__main__':
    pytest.main()
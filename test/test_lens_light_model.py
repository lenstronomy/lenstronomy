__author__ = 'sibirrer'

import pytest
from mock import patch
import numpy as np

from lenstronomy.ImSim.light_model import LensLightModel

class TestLensLightModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_light_type': 'GAUSSIAN',  'lens_type': 'SIS', 'source_type': 'GAUSSIAN'
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'x2_simple': True}
        self.lensLightModel = LensLightModel(self.kwargs_options)
        self.kwargs = {'amp': 1., 'center_x': 0., 'center_y': 0., 'sigma_x': 2., 'sigma_y': 2.}

    def test_surface_brightness(self):
        output = self.lensLightModel.surface_brightness(x=1,y=1, kwargs_lens_light=self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)

if __name__ == '__main__':
    pytest.main()
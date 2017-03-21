__author__ = 'sibirrer'

import pytest
from mock import patch
import numpy as np

from lenstronomy.ImSim.light_model import SourceModel

class TestSourceModel(object):
    """
    tests the source model routines
    """
    @patch("darkskysync.DarkSkySync", autospec=False)
    def setup(self, dss_mock):
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_type': 'SIS', 'source_type': 'GAUSSIAN'
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'x2_simple': True}
        self.sourceModel = SourceModel(self.kwargs_options)
        self.kwargs = {'amp': 1, 'center_x': 0, 'center_y': 0, 'sigma_x': 2, 'sigma_y': 2 }

    def test_surface_brightness(self):
        output = self.sourceModel.surface_brightness(x=1., y=1., **self.kwargs)
        assert output == 0.77880078307140488/(8*np.pi)

if __name__ == '__main__':
    pytest.main()
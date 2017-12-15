__author__ = 'sibirrer'




#from lenstronomy.unit_manager import UnitManager
import numpy as np
import pytest
from mock import patch

import astrofunc.util as util
from lenstronomy.ImSim.lens_model import LensModel


class TestCatalogueCompare(object):

    """
    tests the catalogue mapping routine
    """
    @patch("darkskysync.DarkSkySync", autospec=False)
    def setup(self, dss_mock):
        self.deltaPix = 0.025
        self.x_pos = np.array([-1.25,-0.5,-0.25,1.85])
        self.y_pos = np.array([0.9, 1.375, -1., 0.4])
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_model_list': ['SPEP']
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'x2_simple': True}

        self.lensModel = LensModel(['SPEP'])
        self.kwargs = [{'theta_E': 1./0.8, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.5, 'center_x':0., 'center_y': 0.}] #for SPEP lens

    def test_catalogue_compare(self):
        x_source, y_source = self.lensModel.ray_shooting(self.x_pos, self.y_pos, self.kwargs)
        X2 = util.compare_distance(x_source, y_source)
        assert X2 == 0.41726364831733859

if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'




#from lenstronomy.unit_manager import UnitManager
import pytest
from mock import patch
import numpy as np
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.MCMC.compare import Compare

class TestCatalogueCompare(object):

    """
    tests the catalogue mapping routine
    """
    @patch("darkskysync.DarkSkySync", autospec=False)
    def setup(self, dss_mock):
        kwargs_options = {'X2_compare': 'simple'}
        kwargs_data = {}
        self.compare = Compare(kwargs_options)
        self.deltaPix = 0.025
        self.x_pos = np.array([-1.25,-0.5,-0.25,1.85])
        self.y_pos = np.array([0.9, 1.375, -1., 0.4])
        self.kwargs_options = {'system_name': '', 'data_file': ''
            , 'cosmo_file': '', 'lens_type': 'SPEP', 'source_type': 'GAUSSIAN'
            , 'subgrid_res': 10, 'numPix': 200, 'psf_type': 'GAUSSIAN', 'x2_simple': True}

        self.makeImage = MakeImage(self.kwargs_options, kwargs_data)
        self.kwargs = {'theta_E': 1./0.8, 'gamma': 1.9, 'q': 0.8, 'phi_G': 1.5, 'center_x':0., 'center_y': 0.} #for SPEP lens


    def test_catalogue_compare(self):
        x_source, y_source = self.makeImage.mapping_IS(self.x_pos, self.y_pos, self.kwargs)
        X2 = self.compare.compare_distance(x_source,y_source)
        assert X2 == 0.41726364831733859

if __name__ == '__main__':
    pytest.main()
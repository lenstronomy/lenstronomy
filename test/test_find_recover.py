__author__ = 'sibirrer'
"""
these routines test the ability to find the point sources and how well we can recover them
"""
from lenstronomy.ImSim.make_image import MakeImage
from lenstronomy.Trash.trash import Trash
import numpy as np
import numpy.testing as npt
import pytest
#from lenstronomy.unit_manager import UnitManager

class TestFindRecover(object):

    def setup(self):
        self.kwargs_lens = {'r200': 100, 'center_y_nfw': 0.12897753268824569, 'Rs': 0.36538262531976223, 'center_x_nfw': 0.0058850797857593791, 'rho0': 5.8981737861871126, 'q': 0.76758548426711792, 'center_x': -0.1976795166357318, 'center_y': 0.26130973342152952, 'phi_E': 0.29767869837415178/0.76758548426711792, 'phi_G': 1.3819504031901284, 'gamma': 0.83239856500085108}
        self.kwargs_options = {'X2_type': 'constraint', 'lens_type': 'SPEP_NFW', 'WLS': True, 'num_shapelet_coeffs': 6, 'point_source': True, 'X2_compare': 'standard', 'psf_type': 'pixel', 'lens_light_type': 'DOUBLE_SERSIC', 'error_map': True, 'source_type': 'SERSIC_ELLIPSE', 'shapelet_order': 10, 'shapelets_off': False, 'X2_catalogue': False, 'X2_point_source': False, 'shapelet_beta': 0.2, 'subgrid_res': 2, 'numPix': 150}
        self.kwargs_source = {'n_sersic': 1.829814238850251, 'I0_sersic': 1, 'k_sersic': 2.2651261124736424, 'q': 0.54608543979146251, 'center_x': -0.039639405885285001, 'center_y': 0.73013655722508486, 'phi_G': -1.482282385532782}
        self.kwargs_data = {'deltaPix': 0.049996112860619051, 'exposure_map': None, 'sigma_background': 0.00684717157856, 'mean_background': 0.000430531217717, 'reduced_noise': 1980.0, 'imagePos_x': np.array([-0.27594366, -0.98541128,  1.35378879,  0.26073962]), 'imagePos_y': np.array([-0.86241072,  2.22579356,  1.85606959,  2.34059276])}
        self.makeImage = MakeImage(self.kwargs_options, self.kwargs_data)
        self.trash = Trash(self.makeImage)

        self.kwargs_lens_2 = {'phi_E': 1., 'gamma': 1.9,'q': 0.8, 'phi_G': 0.5, 'center_x': 0.1, 'center_y': -0.1, 'phi_E_spp': 0.1, 'gamma_spp': 1.9, 'center_x_spp': -0.5, 'center_y_spp': 0.5}
        self.kwargs_options_2 = {'lens_type': 'SPEP_SPP', 'source_type': 'GAUSSIAN', 'WLS': True}
        self.makeImage_2 = MakeImage(self.kwargs_options_2,0)
        self.trash_2 = Trash(self.makeImage_2)

    def test_find_recover(self):
        assert 0 == 0
        # initialise lens model and source position
        sourcePos_x = self.kwargs_source['center_x']
        sourcePos_y = self.kwargs_source['center_y']
        deltapix = self.kwargs_data['deltaPix']
        numPix = self.kwargs_options['numPix']
        # compute image positions
        x_pos, y_pos = self.trash.findBrightImage(sourcePos_x, sourcePos_y, deltapix, numPix, magThresh=1., numImage=4, **self.kwargs_lens)
        # fix image positions
        # recover source position
        x_mapped, y_mapped = self.makeImage.mapping_IS(x_pos, y_pos, **self.kwargs_lens)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        # find image positions with new source position
        x_pos_new, y_pos_new = self.trash.findBrightImage(center_x, center_y, deltapix, numPix, magThresh=1., numImage=4, **self.kwargs_lens)
        # compare new/old source and image positions
        npt.assert_almost_equal(x_pos[0], x_pos_new[0], decimal=5)
        npt.assert_almost_equal(x_pos[1], x_pos_new[1], decimal=5)
        npt.assert_almost_equal(x_pos[2], x_pos_new[2], decimal=5)
        npt.assert_almost_equal(x_pos[3], x_pos_new[3], decimal=5)

        npt.assert_almost_equal(y_pos[0], y_pos_new[0], decimal=5)
        npt.assert_almost_equal(y_pos[1], y_pos_new[1], decimal=5)
        npt.assert_almost_equal(y_pos[2], y_pos_new[2], decimal=5)
        npt.assert_almost_equal(y_pos[3], y_pos_new[3], decimal=5)

        npt.assert_almost_equal(center_x, sourcePos_x, decimal=6)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=6)

    def test_find_recover_SPP(self):
        sourcePos_x = 0.1
        sourcePos_y = -0.1
        deltapix = self.kwargs_data['deltaPix']
        numPix = self.kwargs_options['numPix']
        # compute image positions
        x_pos, y_pos = self.trash_2.findBrightImage(sourcePos_x, sourcePos_y, deltapix, numPix, magThresh=1., numImage=4, **self.kwargs_lens_2)
        # fix image positions
        # recover source position
        x_mapped, y_mapped = self.makeImage_2.mapping_IS(x_pos, y_pos, **self.kwargs_lens_2)
        center_x, center_y = np.mean(x_mapped), np.mean(y_mapped)
        npt.assert_almost_equal(center_x, sourcePos_x, decimal=5)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=5)
        # find image positions with new source position
        x_pos_new, y_pos_new = self.trash_2.findBrightImage(center_x, center_y, deltapix, numPix, magThresh=1., numImage=4, **self.kwargs_lens_2)
        # compare new/old source and image positions
        npt.assert_almost_equal(x_pos[0], x_pos_new[0], decimal=5)
        npt.assert_almost_equal(x_pos[1], x_pos_new[1], decimal=5)
        npt.assert_almost_equal(x_pos[2], x_pos_new[2], decimal=5)
        npt.assert_almost_equal(x_pos[3], x_pos_new[3], decimal=5)

        npt.assert_almost_equal(y_pos[0], y_pos_new[0], decimal=5)
        npt.assert_almost_equal(y_pos[1], y_pos_new[1], decimal=5)
        npt.assert_almost_equal(y_pos[2], y_pos_new[2], decimal=5)
        npt.assert_almost_equal(y_pos[3], y_pos_new[3], decimal=5)

        npt.assert_almost_equal(center_x, sourcePos_x, decimal=6)
        npt.assert_almost_equal(center_y, sourcePos_y, decimal=6)

    # def test_find_image(self):
    #     kwargs_lens = {'r200': 100, 'center_y_nfw': 0.024301278425072279, 'Rs': 0.31825360665713964, 'center_x_nfw': 0.12870026870830353, 'rho0': 8.9149918802925594, 'q': 0.74383545496446302, 'center_x': -0.30004691263957234, 'center_y': 0.42070715820673332, 'phi_E': 0.26574914582287257, 'phi_G': 1.3241065135638992, 'gamma': 1.6074909699843223}
    #     kwargs_source = {'n_sersic': 2.0032820795643604, 'I0_sersic': 1, 'k_sersic': 2.4737392073696869, 'q': 0.53616501388254001, 'center_x': -0.044151764654922855, 'center_y': 0.7880703877671551, 'phi_G': -1.4617423110488805}
    #     sourcePos_x = kwargs_source['center_x']
    #     sourcePos_y = kwargs_source['center_y']
    #     deltapix = self.kwargs_data['deltaPix']
    #     numPix = self.kwargs_options['numPix']
    #
    #     x_pos, y_pos = self.makeImage.findImage(sourcePos_x, sourcePos_y, deltapix, numPix, **kwargs_lens)
    #     print x_pos, y_pos
    #     assert 1==0


if __name__ == '__main__':
    pytest.main()
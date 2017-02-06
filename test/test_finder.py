__author__ = 'sibirrer'

import numpy as np
import numpy.testing as npt
import pytest

# from lenstronomy.ImSim.make_image import MakeImage
# from lenstronomy.Workflow_old.position_finder import PositionFinder
#
# class TestFind(object):
#
#     def setup(self):
#         kwargs_options = {'X2_type': 'catalogue', 'lens_type': 'SPEP', 'WLS': False, 'point_source': True, 'X2_compare': 'standard', 'psf_type': 'GAUSSIAN', 'lens_light_type': 'DOUBLE_SERSIC', 'error_map': True, 'source_type': 'SERSIC_ELLIPSE', 'shapelet_order': 10, 'shapelets_off': False, 'X2_catalogue': False, 'X2_point_source': False, 'shapelet_beta': 0.2, 'subgrid_res': 2, 'numPix': 150}
#         kwargs_data = {'deltaPix': 0.049996112860619051, 'exposure_map': None, 'sigma_background': 0.00684717157856, 'mean_background': 0.000430531217717, 'reduced_noise': 1980.0
#             , 'image_data': [0]}
#
#
#         kwargs_lens_clump_init = {'Rs': 0.1,'rho0': 1, 'r200':100, 'center_x_nfw': 0, 'center_y_nfw': 0,
#                                 'phi_E_sis': 0.1, 'center_x_sis': 0, 'center_y_sis': 0
#                                 , 'phi_E_spp': 0.1, 'gamma_spp': 2, 'center_x_spp': 0, 'center_y_spp': 0}
#         kwargs_lens_clump_sigma_init = {'Rs_sigma':1, 'rho0_sigma': 1, 'r200_sigma': 1
#                                 , 'center_x_nfw_sigma': 2, 'center_y_nfw_sigma': 2,
#                                 'phi_E_sis_sigma': 1, 'center_x_sis_sigma': 2, 'center_y_sis_sigma': 2
#                                 , 'phi_E_spp_sigma': 1, 'gamma_spp_sigma': 0.1, 'center_x_spp_sigma': 2, 'center_y_spp_sigma': 2}
#         kwargs_lens_clump_sigma_weak = {'Rs_sigma':0.1, 'rho0_sigma': 0.1, 'r200_sigma': 0.1
#                                 , 'center_x_nfw_sigma': 0.1, 'center_y_nfw_sigma': 0.1,
#                                 'phi_E_sis_sigma': 0.1, 'center_x_sis_sigma': 0.1, 'center_y_sis_sigma': 0.1
#                                 , 'phi_E_spp_sigma': 0.1, 'gamma_spp_sigma': 0.1, 'center_x_spp_sigma': 0.1, 'center_y_spp_sigma': 0.1}
#         kwargs_lens_clump_sigma_constraint = {'Rs_sigma':0.01, 'rho0_sigma': 0.01, 'r200_sigma': 0.01
#                                 , 'center_x_nfw_sigma': 0.01, 'center_y_nfw_sigma': 0.01,
#                                 'phi_E_sis_sigma': 0.01, 'center_x_sis_sigma': 0.01, 'center_y_sis_sigma': 0.01
#                                 , 'phi_E_spp_sigma': 0.01, 'gamma_spp_sigma': 0.01, 'center_x_spp_sigma': 0.01, 'center_y_spp_sigma': 0.01}
#
#         kwargs_init = [{'phi_E': 1.,'q':0.9, 'gamma': 1.9
#                               ,'phi_G':1.5, 'center_x':0., 'center_y':0.},
#                         {'I0_sersic':10. ,'center_x':0, 'center_y':0, 'k_sersic': 10, 'n_sersic': 4, 'phi_G':0, 'q':0.8},
#                         {'sigma': 1},
#                         {'I0_sersic':3. ,'center_x':1., 'center_y':1., 'k_sersic': 3, 'n_sersic': 1,
#                          'phi_G':0.5, 'q':0.3, 'I0_2':1., 'k_2': 3, 'n_2': 1, 'center_x_2':1., 'center_y_2':1.},
#                         {'point_amp': 1., 'x_pos': np.array([0.588, 0.618, 0, -2.517]) +1.7, 'y_pos': np.array([1.120, 2.307, 0, 1.998, ]) -1.9},
#                         kwargs_lens_clump_init]
#         kwargs_sigma_init = [{'phi_E_sigma': .2,'q_sigma':0.3, 'gamma_sigma': 0.3
#                               ,'phi_G_sigma':0.5, 'center_x_sigma':0.5, 'center_y_sigma':0.5},
#                         {'I0_sersic_sigma':3. ,'center_x_sigma':1., 'center_y_sigma':1., 'k_sersic_sigma': 3, 'n_sersic_sigma': 1,
#                          'phi_G_sigma':0.5, 'q_sigma':0.3},
#                         {},
#                         {'I0_sersic_sigma':3. ,'center_x_sigma':1., 'center_y_sigma':1., 'k_sersic_sigma': 3, 'n_sersic_sigma': 1,
#                          'phi_G_sigma':0.5, 'q_sigma':0.3, 'I0_2_sigma':1., 'k_2_sigma': 3, 'n_2_sigma': 1, 'center_x_2_sigma':1., 'center_y_2_sigma':1.},
#                         {'point_amp_sigma': 1, 'pos_sigma': 2},
#                         kwargs_lens_clump_sigma_init]
#         kwargs_sigma_weak = [{'phi_E_sigma': 0.1,'q_sigma':0.1, 'gamma_sigma': 0.1
#                               ,'phi_G_sigma':0.1, 'center_x_sigma':0.1, 'center_y_sigma':0.1},
#                         {'I0_sersic_sigma':0.1 ,'center_x_sigma':0.1, 'center_y_sigma':0.1, 'k_sersic_sigma': 0.1, 'n_sersic_sigma': 0.1,
#                          'phi_G_sigma':0.5, 'q_sigma':0.3},
#                         {},
#                         {'I0_sersic_sigma': 0.1,'center_x_sigma':0.1, 'center_y_sigma':0.1, 'k_sersic_sigma': 0.1, 'n_sersic_sigma': 0.1,
#                          'phi_G_sigma':0.1, 'q_sigma':0.1, 'I0_2_sigma':0.1, 'k_2_sigma': 0.1, 'n_2_sigma': 0.1, 'center_x_2_sigma':0.1, 'center_y_2_sigma':0.1},
#                         {'point_amp_sigma': 0.1, 'pos_sigma': 0.01},
#                         kwargs_lens_clump_sigma_weak]
#         kwargs_sigma_constraint = [{'phi_E_sigma': 0.01,'q_sigma':0.01, 'gamma_sigma': 0.01
#                               ,'phi_G_sigma':0.01, 'center_x_sigma':0.01, 'center_y_sigma':0.01},
#                         {'I0_sersic_sigma':.01 ,'center_x_sigma':0.01, 'center_y_sigma':0.01, 'k_sersic_sigma': 0.01, 'n_sersic_sigma': 0.01,
#                          'phi_G_sigma':0.01, 'q_sigma':0.01},
#                         {},
#                         {'I0_sersic_sigma':0.01 ,'center_x_sigma':0.01, 'center_y_sigma':0.01, 'k_sersic_sigma': 0.01, 'n_sersic_sigma': 0.01,
#                          'phi_G_sigma':0.01, 'q_sigma':0.01, 'I0_2_sigma':0.01, 'k_2_sigma': 0.01, 'n_2_sigma': 0.01, 'center_x_2_sigma':0.01, 'center_y_2_sigma':0.01},
#                         {'point_amp_sigma': 0.01, 'pos_sigma': 0.001},
#                         kwargs_lens_clump_sigma_constraint]
#
#         self.positionFinder = PositionFinder(kwargs_data, kwargs_options, kwargs_init, kwargs_sigma_init, kwargs_sigma_weak, kwargs_sigma_constraint)
#
#
#     def test_find_catalogue(self):
#         # initialise lens model and source position
#         lens_result, source_result, psf_result, lens_light_result, else_result, chain = self.positionFinder.find_param_catalogue(n_particles=100,n_iterations=500)
#         print lens_result, source_result, psf_result, lens_light_result, else_result
#         npt.assert_almost_equal(lens_result['phi_G'], -0.31419438918067533, decimal=2)
#         npt.assert_almost_equal(lens_result['center_x'], -0.053882972177104492, decimal=2)
#         npt.assert_almost_equal(lens_result['center_y'], -0.15590695491668566, decimal=2)
#         assert 1 == 0


if __name__ == '__main__':
    pytest.main()
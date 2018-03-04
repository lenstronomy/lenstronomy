__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.LensModel.lens_param import LensParam


class TestParam(object):

    def setup(self):
        self.lens_model_list = ['SPEP', 'SHEAR', 'FLEXION', 'GAUSSIAN', 'SIS', 'SIS_TRUNCATED', 'SPP',
                                'NFW', 'TNFW', 'NFW_ELLIPSE', 'SERSIC', 'SERSIC_ELLIPSE', 'SERSIC_DOUBLE', 'COMPOSITE',
                                'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST', 'HERNQUIST_ELLIPSE', 'GAUSSIAN',
                                'GAUSSIAN_KAPPA', 'INTERPOL_SCALED', 'SHAPELETS_POLAR', 'SHAPELETS_CART', 'DIPOLE'
                                ]
        self.kwargs = [
            {'theta_E': 1., 'gamma': 2, 'q': 0.7, 'phi_G': 0.5, 'center_x': 0, 'center_y': 0},  # 'SPEP
            {'e1': 0.1, 'e2': 0.1},  # EXTERNAL_SHEAR
            {'g1': 0.01, 'g2': 0.01, 'g3': -0.01, 'g4': 0},  # 'FLEXION'
            {'amp': 1., 'sigma_x': 1, 'sigma_y': 1., 'center_x': 0, 'center_y': 0},  # 'GAUSSIAN'
            {'theta_E': 1., 'center_x': 0, 'center_y': 0},  # 'SIS
            {'theta_E': 1, 'r_trunc': 2., 'center_x': 0, 'center_y': 0},  # 'SIS_TRUNCATED'
            {'theta_E': 1, 'gamma': 2, 'center_x': 0, 'center_y': 0},  # 'SPP'
            {'Rs': 1, 'theta_Rs': 0.1, 'center_x': 0, 'center_y': 0},  # 'NFW'
            {'Rs': 1, 'theta_Rs': 0.1, 'r_trunc': 10, 'center_x': 0, 'center_y': 0},  # 'TNFW'
            {'Rs': 1, 'q': 0.8, 'phi_G': 0.5, 'theta_Rs': 0.1, 'center_x': 0, 'center_y': 0},  # 'NFW_ELLIPSE
            {'r_eff': 1, 'n_sersic': 2, 'k_eff': 0.5, 'center_x': 0, 'center_y': 0},  # 'SERSIC'
            {'r_eff': 1, 'n_sersic': 2, 'k_eff': 0.5, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0.5},  # 'SERSIC_ELLIPSE'
            {'r_eff': 1, 'n_sersic': 2, 'k_eff': 0.5, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 0.5, 'q_2': 0.8,
             'phi_G_2': 0.5, 'n_2': 1., 'R_2': 0.1, 'flux_ratio': 0.5},  # 'SERSIC_DOUBLE'
            {'theta_E': 1, 'mass_light': 0.5, 'Rs': 1, 'q': 0.7, 'phi_G': 0, 'n_sersic': 1, 'r_eff': 0.6, 'q_s': 0.9,
             'phi_G_s': 0.8, 'center_x': 0, 'center_y': 0},  # 'COMPOSITE'
            {'sigma0': 0.5, 'Ra': 0.7, 'Rs': 0.2, 'center_x': 0, 'center_y': 0},  # 'PJAFFE'
            {'sigma0': 0.5, 'Ra': 0.7, 'Rs': 0.2, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 1.},  # 'PJAFFE_ELLIPSE'
            {'sigma0': 0.5, 'Rs': 0.5, 'center_x': 0, 'center_y': 0},  # 'HERNQUIST'
            {'sigma0': 0.5, 'Rs': 0.5, 'center_x': 0, 'center_y': 0, 'q': 0.8, 'phi_G': 1.},  # 'HERNQUIST_ELLIPSE'
            {'amp': 1, 'sigma_x': 0.5, 'sigma_y': 0.5, 'center_x': 0, 'center_y': 0},  # 'GAUSSIAN'
            {'amp': 1, 'sigma': 0.5, 'center_x': 0, 'center_y': 0},  # 'GAUSSIAN_KAPPA'
            {'scale_factor': 1, 'grid_interp_x': None, 'grid_interp_y': None, 'f_x': None, 'f_y': None},  # 'INTERPOL_SCALED'
            {'coeffs': [1, 1], 'beta': 1., 'center_x': 0, 'center_y': 0},  # 'SHAPELETS_POLAR'
            {'coeffs': [1, 1], 'beta': 1., 'center_x': 0, 'center_y': 0},  # 'SHAPELETS_CART'
            {'coupling': 1, 'phi_dipole': 1, 'center_x': 0, 'center_y': 0},  # 'DIPOLE'
            ]
        self.kwargs_sigma = [
            {'theta_E_sigma': 1., 'gamma_sigma': 2, 'ellipse_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SPEP
            {'shear_sigma': 0.1},  # EXTERNAL_SHEAR
            {'flexion_sigma': 0.01},  # 'FLEXION'
            {'amp_sigma': 1., 'sigma_x_sigma': 1, 'sigma_y_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'GAUSSIAN'
            {'theta_E_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SIS
            {'theta_E_sigma': 1, 'r_trunc_sigma': 2., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SIS_TRUNCATED'
            {'theta_E_sigma': 1, 'gamma_sigma': 2, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SPP'
            {'Rs_sigma': 1, 'theta_Rs_sigma': 0.1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'NFW'
            {'Rs_sigma': 1, 'theta_Rs_sigma': 0.1, 'r_trunc_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'TNFW'
            {'Rs_sigma': 1, 'ellipse_sigma': 0.1, 'theta_Rs_sigma': 0.1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'NFW_ELLIPSE
            {'r_eff_sigma': 1, 'n_sersic_sigma': 2, 'k_eff_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SERSIC'
            {'r_eff_sigma': 1, 'n_sersic_sigma': 2, 'k_eff_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'ellipse_sigma': 0.1},
            # 'SERSIC_ELLIPSE'
            {'r_eff_sigma': 1, 'n_sersic_sigma': 2, 'k_eff_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'ellipse_sigma': 0.1
                , 'n_2_sigma': 1., 'R_2_sigma': 0.1, 'flux_ratio_sigma': 0.5},  # 'SERSIC_DOUBLE'
            {'theta_E_sigma': 1, 'mass_light_sigma': 0.5, 'Rs_sigma': 1, 'ellipse_sigma': 0.1, 'ellipse_s_sigma': 0.1, 'n_sersic_sigma': 1, 'r_eff_sigma': 0.6, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'COMPOSITE'
            {'sigma0_sigma': 0.5, 'Ra_sigma': 0.7, 'Rs_sigma': 0.2, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'PJAFFE'
            {'sigma0_sigma': 0.5, 'Ra_sigma': 0.7, 'Rs_sigma': 0.2, 'center_x_sigma': 0, 'center_y_sigma': 00, 'ellipse_sigma': 0.1},
            # 'PJAFFE_ELLIPSE'
            {'sigma0_sigma': 0.5, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'HERNQUIST'
            {'sigma0_sigma': 0.5, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'ellipse_sigma': 0.1},  # 'HERNQUIST_ELLIPSE'
            {'amp_sigma': 1, 'sigma_x_sigma': 0.5, 'sigma_y_sigma': 0.1, 'center_x_sigma': 0, 'center_y_sigma': 0, 'center_y': 0},  # 'GAUSSIAN'
            {'amp_sigma': 1, 'sigma_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'GAUSSIAN_KAPPA'
            {'scale_factor_sigma': 1},
            # 'INTERPOL_SCALED'
            {'coeffs_sigma': 0.1, 'beta_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS_POLAR'
            {'coeffs_sigma': 0.1, 'beta_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS_CART'
            {'coupling_sigma': 1, 'phi_dipole_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'DIPOLE'
        ]
        self.kwargs_fixed = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
                             {'grid_interp_x': None, 'grid_interp_y': None, 'f_x': None, 'f_y': None},
                             {}, {}, {'phi_dipole': 1.}]
        self.kwargs_mean = []
        for i in range(len(self.lens_model_list)):
            kwargs_mean_k = self.kwargs[i].copy()
            kwargs_mean_k.update(self.kwargs_sigma[i])
            self.kwargs_mean.append(kwargs_mean_k)
        self.param = LensParam(lens_model_list=self.lens_model_list,
                               kwargs_fixed=self.kwargs_fixed, num_images=2, solver_type='NONE', num_shapelet_lens=2)
        self.param_fixed = LensParam(lens_model_list=self.lens_model_list,
                               kwargs_fixed=self.kwargs, num_images=2, solver_type='NONE', num_shapelet_lens=2)

    def test_get_setParams(self):
        args = self.param.setParams(self.kwargs)
        kwargs_new, _ = self.param.getParams(args, i=0)
        args_new = self.param.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_fixed.setParams(self.kwargs)
        kwargs_new, _ = self.param_fixed.getParams(args, i=0)
        args_new = self.param_fixed.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_param_init(self):
        mean, sigma = self.param.param_init(self.kwargs_mean)
        assert mean[0] == 1

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 119


if __name__ == '__main__':
    pytest.main()

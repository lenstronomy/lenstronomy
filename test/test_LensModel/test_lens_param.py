__author__ = 'sibirrer'

import pytest
import numpy.testing as npt
from lenstronomy.LensModel.lens_param import LensParam


class TestParam(object):

    def setup(self):
        self.lens_model_list = ['SPEP',
                                'INTERPOL_SCALED',
                                'SHAPELETS_CART',
                                'MULTI_GAUSSIAN_KAPPA'
                                ]
        self.kwargs = [
            {'theta_E': 1., 'gamma': 2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0},  # 'SPEP
            {'scale_factor': 1, 'grid_interp_x': None, 'grid_interp_y': None, 'f_x': None, 'f_y': None},  # 'INTERPOL_SCALED'
            {'coeffs': [1, 1, 1, 1, 1, 1], 'beta': 1., 'center_x': 0, 'center_y': 0},  # 'SHAPELETS_CART'
            {'amp': [1, 2], 'sigma': [0.5, 1], 'center_x': 0, 'center_y': 0, 'scale_factor': 1},  # 'MULTI_GAUSSIAN_KAPPA'
            ]
        self.kwargs_sigma = [
            {'theta_E_sigma': 1., 'gamma_sigma': 2, 'e1_sigma': 0.1, 'e2_sigma': 0.1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SPEP
            {'scale_factor_sigma': 1},  # 'INTERPOL_SCALED'
            {'coeffs_sigma': 0.1, 'beta_sigma': 1., 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS_CART'
            {'amp_sigma': [1, 1], 'sigma_sigma': [1, 1], 'center_x_sigma': 0, 'center_y_sigma': 0, 'scale_factor_sigma': 1},
        ]
        self.kwargs_fixed = [{},
                             {'grid_interp_x': None, 'grid_interp_y': None, 'f_x': None, 'f_y': None},
                             {},
                             {'sigma': [1, 2]}
                             ]
        self.kwargs_mean = []
        for i in range(len(self.lens_model_list)):
            kwargs_mean_k = self.kwargs[i].copy()
            kwargs_mean_k.update(self.kwargs_sigma[i])
            self.kwargs_mean.append(kwargs_mean_k)
        self.param = LensParam(lens_model_list=self.lens_model_list,
                               kwargs_fixed=self.kwargs_fixed, num_images=2, solver_type='SHAPELETS', num_shapelet_lens=6)
        self.param_fixed = LensParam(lens_model_list=self.lens_model_list,
                               kwargs_fixed=self.kwargs, num_images=4, solver_type='NONE', num_shapelet_lens=6)

    def test_get_setParams(self):
        print(self.kwargs, 'kwargs')
        args = self.param.setParams(self.kwargs)
        print(args, 'args')
        kwargs_new, _ = self.param.getParams(args, i=0)
        print(kwargs_new, 'kwargs_new')
        args_new = self.param.setParams(kwargs_new)
        print(args_new, 'args_new')
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_fixed.setParams(self.kwargs)
        kwargs_new, _ = self.param_fixed.getParams(args, i=0)
        args_new = self.param_fixed.setParams(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_param_name_list(self):
        lens_model_list = ['SHIFT', 'FLEXION', 'SIS_TRUNCATED', 'SERSIC',
                           'SERSIC_ELLIPSE_POTENTIAL', 'SERSIC_ELLIPSE_KAPPA',
                           'PJAFFE', 'PJAFFE_ELLIPSE', 'HERNQUIST_ELLIPSE', 'INTERPOL', 'INTERPOL_SCALED',
                           'SHAPELETS_POLAR', 'DIPOLE',
                           'GAUSSIAN_ELLIPSE_KAPPA', 'SERSIC_ELLIPSE_KAPPA',
                           'SERSIC_ELLIPSE_GAUSS_DEC', 'NFW_ELLIPSE_GAUSS_DEC',
                           'CTNFW_GAUSS_DEC',
                           'GAUSSIAN_ELLIPSE_POTENTIAL',
                           'MULTI_GAUSSIAN_KAPPA',
                           'MULTI_GAUSSIAN_KAPPA_ELLIPSE']
        lensParam = LensParam(lens_model_list, kwargs_fixed=None)
        param_name_list = lensParam._param_name_list
        assert len(lens_model_list) == len(param_name_list)

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == 20

    def test_shapelet_solver(self):
        lens_model_list = ['SHAPELETS_CART']
        lensParam = LensParam(lens_model_list, kwargs_fixed=[{}], num_images=2, solver_type='SHAPELETS',
                              num_shapelet_lens=8)
        kwargs_lens = [{'beta': 1, 'coeffs': [0, 1, 2, 3, 4, 5, 5, 7], 'center_x':0, 'center_y': 0}]
        args = lensParam.setParams(kwargs_lens)
        kwargs_out, i = lensParam.getParams(args, i=0)
        assert kwargs_out[0]['coeffs'][1] == 0
        assert kwargs_out[0]['beta'] == kwargs_lens[0]['beta']
        num, param_list = lensParam.num_param()
        assert num == 8

        lensParam = LensParam(lens_model_list, kwargs_fixed=[{}], num_images=4, solver_type='SHAPELETS',
                              num_shapelet_lens=8)
        kwargs_lens = [{'beta': 1, 'coeffs': [0, 1, 2, 3, 4, 5, 5, 7], 'center_x': 0, 'center_y': 0}]
        args = lensParam.setParams(kwargs_lens)
        kwargs_out, i = lensParam.getParams(args, i=0)
        assert kwargs_out[0]['coeffs'][5] == 0
        assert kwargs_out[0]['beta'] == kwargs_lens[0]['beta']
        num, param_list = lensParam.num_param()
        assert num == 5


if __name__ == '__main__':
    pytest.main()

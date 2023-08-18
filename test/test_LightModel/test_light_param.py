__author__ = "sibirrer"

import pytest
import numpy as np
import numpy.testing as npt
import unittest
from lenstronomy.LightModel.light_param import LightParam


class TestParam(object):
    def setup_method(self):
        self.light_model_list = [
            "GAUSSIAN",
            "MULTI_GAUSSIAN",
            "SERSIC",
            "SERSIC_ELLIPSE",
            "CORE_SERSIC",
            "SHAPELETS",
            "HERNQUIST",
            "HERNQUIST_ELLIPSE",
            "PJAFFE",
            "PJAFFE_ELLIPSE",
            "UNIFORM",
            "SHAPELETS",
            "SHAPELETS_POLAR_EXP",
            "SLIT_STARLETS",
        ]
        self.kwargs = [
            {"amp": 1.0, "sigma": 1, "center_x": 0, "center_y": 0},  # 'GAUSSIAN'
            {
                "amp": [1.0, 2],
                "sigma": [1, 3],
                "center_x": 0,
                "center_y": 0,
            },  # 'MULTI_GAUSSIAN'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "n_sersic": 1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC_ELLIPSE'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "Rb": 0.1,
                "gamma": 2.0,
                "n_sersic": 1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0,
                "center_y": 0,
            },
            # 'CORE_SERSIC'
            {
                "amp": [1, 1, 1],
                "beta": 0.5,
                "n_max": 1,
                "center_x": 0,
                "center_y": 0,
            },  # 'SHAPELETS'
            {"amp": 1, "Rs": 0.5, "center_x": 0, "center_y": 0},  # 'HERNQUIST'
            {
                "amp": 1,
                "Rs": 0.5,
                "center_x": 0,
                "center_y": 0,
                "e1": 0.1,
                "e2": 0.1,
            },  # 'HERNQUIST_ELLIPSE'
            {"amp": 1, "Ra": 1, "Rs": 0.5, "center_x": 0, "center_y": 0},  # 'PJAFFE'
            {
                "amp": 1,
                "Ra": 1,
                "Rs": 0.5,
                "center_x": 0,
                "center_y": 0,
                "e1": 0.1,
                "e2": 0.1,
            },  # 'PJAFFE_ELLIPSE'
            {"amp": 1},  # 'UNIFORM'
            {
                "amp": [1],
                "beta": 1,
                "n_max": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'SHAPELETS'
            {
                "amp": [1],
                "beta": 1,
                "n_max": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'SHAPELETS_POLAR_EXP'
            {
                "amp": np.ones((3 * 20**2,)),
                "n_scales": 3,
                "n_pixels": 20**2,
                "scale": 0.05,
                "center_x": 0,
                "center_y": 0,
            },  # 'SLIT_STARLETS'
        ]
        # self.kwargs_sigma = [
        #     {'amp_sigma': 1., 'sigma_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},
        #     # 'GAUSSIAN'
        #     {'amp_sigma': [1., 1.], 'sigma_sigma': [1, 1], 'center_x_sigma': 0, 'center_y_sigma': 0},
        #     # 'MULTI_GAUSSIAN'
        #     {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1, 'center_y_sigma': 1},  # 'SERSIC'
        #     {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
        #      'center_y_sigma': 1, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'SERSIC_ELLIPSE'
        #     {'amp_sigma': 1, 'R_sersic_sigma': 0.5, 'n_sersic_sigma': 1, 'center_x_sigma': 1,
        #      'center_y_sigma': 1, 'e1_sigma': 0.1, 'e2_sigma': 0.1, 'Re_sigma': 0.01, 'gamma_sigma': 0.1},  # 'CORE_SERSIC'
        #     {'amp_sigma': [1, 1, 1], 'beta_sigma': 0.1, 'n_max_sigma': 1, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'SHAPELETS'
        #     {'amp_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'HERNQUIST'
        #     {'amp_sigma': 1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'HERNQUIST_ELLIPSE'
        #     {'amp_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0},  # 'PJAFFE'
        #     {'amp_sigma': 1, 'Ra_sigma': 0.1, 'Rs_sigma': 0.5, 'center_x_sigma': 0, 'center_y_sigma': 0, 'e1_sigma': 0.1, 'e2_sigma': 0.1},  # 'PJAFFE'
        #     {'amp_sigma': 0.1},  # 'UNIFORM'
        # ]
        self.kwargs_fixed = [
            {},
            {"sigma": [1, 3]},
            {},
            {},
            {},
            {"n_max": 1},
            {},
            {},
            {},
            {},
            {},
            {"n_max": 0},
            {"n_max": 0},
            {
                "n_scales": 3,
                "n_pixels": 20**2,
                "scale": 0.05,
                "center_x": 0,
                "center_y": 0,
            },
        ]
        self.kwargs_fixed_linear = [
            {},
            {"sigma": [1, 3]},
            {},
            {},
            {},
            {"n_max": 1},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {
                "n_scales": 3,
                "n_pixels": 20**2,
                "scale": 0.05,
                "center_x": 0,
                "center_y": 0,
            },
        ]
        # self.kwargs_mean = []
        # for i in range(len(self.light_model_list)):
        #     kwargs_mean_k = self.kwargs[i].copy()
        #     #kwargs_mean_k.update(self.kwargs_sigma[i])
        #     self.kwargs_mean.append(kwargs_mean_k)
        self.param = LightParam(
            light_model_list=self.light_model_list,
            kwargs_fixed=self.kwargs_fixed,
            param_type="source_light",
            linear_solver=False,
        )
        self.param_linear = LightParam(
            light_model_list=self.light_model_list,
            kwargs_fixed=self.kwargs_fixed_linear,
            param_type="source_light",
            linear_solver=True,
        )
        self.param_fixed = LightParam(
            light_model_list=self.light_model_list,
            kwargs_fixed=self.kwargs,
            param_type="source_light",
            linear_solver=False,
        )

    def test_get_setParams(self):
        args = self.param.set_params(self.kwargs)
        kwargs_new, _ = self.param.get_params(args, i=0)
        args_new = self.param.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_fixed.set_params(self.kwargs)
        kwargs_new, _ = self.param_fixed.get_params(args, i=0)
        args_new = self.param_fixed.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

        args = self.param_linear.set_params(self.kwargs)
        kwargs_new, _ = self.param_linear.get_params(args, i=0)
        args_new = self.param_linear.set_params(kwargs_new)
        for k in range(len(args)):
            npt.assert_almost_equal(args[k], args_new[k], decimal=8)

    def test_num_params(self):
        num, list = self.param.num_param()
        assert num == (66 + 1200)

    def test_param_name_list(self):
        param_name_list = self.param.param_name_list
        assert param_name_list[0][0] == "amp"

    def test_num_param_linear(self):
        kwargs_fixed = [
            {},
            {"sigma": [1, 3]},
            {},
            {},
            {},
            {"n_max": 1},
            {},
            {},
            {},
            {},
            {},
            {"n_max": 0},
            {"n_max": 0},
            {
                "n_scales": 3,
                "n_pixels": 20**2,
                "scale": 0.05,
                "center_x": 0,
                "center_y": 0,
            },
        ]
        param = LightParam(
            light_model_list=self.light_model_list,
            kwargs_fixed=kwargs_fixed,
            param_type="source_light",
            linear_solver=True,
        )

        num = param.num_param_linear()
        assert num == (16 + 1200)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            lighModel = LightParam(light_model_list=["WRONG"], kwargs_fixed=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["MULTI_GAUSSIAN"], kwargs_fixed=[{}]
            )
            lighModel.set_params(kwargs_list=[{"amp": 1, "sigma": 1}])
        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["SHAPELETS"], kwargs_fixed=[{}], linear_solver=False
            )
            lighModel.num_param()
        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["SHAPELETS"], kwargs_fixed=[{}], linear_solver=False
            )
            lighModel.get_params(args=[], i=0)
        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["MULTI_GAUSSIAN"],
                kwargs_fixed=[{}],
                linear_solver=False,
            )
            lighModel.get_params(args=[1, 1, 1, 1], i=0)
        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{}],
                linear_solver=False,
            )
            lighModel.get_params(args=[1], i=0)
        with self.assertRaises(ValueError):
            # no fixed params provided
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{}],
                linear_solver=False,
            )
            lighModel.set_params(kwargs_list=[{"amp": np.ones((3 * 20**2))}])
        with self.assertRaises(ValueError):
            # missing fixed params
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{"n_scales": 3}],
                linear_solver=False,
            )
            lighModel.set_params(kwargs_list=[{"amp": np.ones((3 * 20**2))}])
        with self.assertRaises(ValueError):
            # missing fixed params
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{"n_scales": 3}],
                linear_solver=False,
            )
            lighModel.num_param()
        with self.assertRaises(ValueError):
            # missing fixed params
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{"amp": np.ones((3 * 20**2))}],
                linear_solver=False,
            )
            lighModel.set_params([{"n_scales": 3}])

        with self.assertRaises(ValueError):
            # missing fixed params 'n_pixels'
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{}],
                linear_solver=False,
            )
            lighModel.set_params([{"n_scales": 3}])

            # missing fixed params 'n_scales'
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{}],
                linear_solver=False,
            )
            lighModel.set_params([{"n_pixels": 3}])

        with self.assertRaises(ValueError):
            lighModel = LightParam(
                light_model_list=["SLIT_STARLETS"],
                kwargs_fixed=[{"amp": np.ones((3 * 20**2))}],
                linear_solver=False,
            )
            lighModel.set_params([{"n_scales": 3}])


if __name__ == "__main__":
    pytest.main()

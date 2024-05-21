__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
import unittest
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.gaussian import Gaussian


class TestLightModel(object):
    """Tests the source model routines."""

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
            "POWER_LAW",
            "NIE",
            "INTERPOL",
            "SHAPELETS_POLAR_EXP",
            "ELLIPSOID",
            "LINE_PROFILE",
        ]
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        self.kwargs = [
            {"amp": 1.0, "sigma": 1.0, "center_x": 0, "center_y": 0},  # 'GAUSSIAN'
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
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC_ELLIPSE'
            {
                "amp": 1,
                "R_sersic": 0.5,
                "Rb": 0.1,
                "gamma": 2.0,
                "n_sersic": 1,
                "e1": e1,
                "e2": e2,
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
                "e1": e1,
                "e2": e2,
            },  # 'HERNQUIST_ELLIPSE'
            {"amp": 1, "Ra": 1, "Rs": 0.5, "center_x": 0, "center_y": 0},  # 'PJAFFE'
            {
                "amp": 1,
                "Ra": 1,
                "Rs": 0.5,
                "center_x": 0,
                "center_y": 0,
                "e1": e1,
                "e2": e2,
            },  # 'PJAFFE_ELLIPSE'
            {"amp": 1},  # 'UNIFORM'
            {
                "amp": 1.0,
                "gamma": 2.0,
                "e1": e1,
                "e2": e2,
                "center_x": 0,
                "center_y": 0,
            },  # 'POWER_LAW'
            {
                "amp": 0.001,
                "e1": 0,
                "e2": 1.0,
                "center_x": 0,
                "center_y": 0,
                "s_scale": 1.0,
            },  # 'NIE'
            {
                "image": np.zeros((20, 5)),
                "scale": 1,
                "phi_G": 0,
                "center_x": 0,
                "center_y": 0,
            },
            {"amp": [1], "n_max": 0, "beta": 1, "center_x": 0, "center_y": 0},
            {
                "amp": 1,
                "radius": 1.0,
                "e1": 0,
                "e2": 0.1,
                "center_x": 0,
                "center_y": 0,
            },  # 'ELLIPSOID'
            {
                "amp": 1,
                "length": 1.0,
                "width": 0.01,
                "angle": 57,
                "start_x": 0,
                "start_y": 0,
            },  # 'LINE_PROFILE'
        ]

        self.LightModel = LightModel(
            light_model_list=self.light_model_list, sersic_major_axis=False
        )

    def test_init(self):
        model_list = [
            "CORE_SERSIC",
            "SHAPELETS",
            "SHAPELETS_ELLIPSE",
            "SHAPELETS_POLAR",
            "SHAPELETS_POLAR_EXP",
            "UNIFORM",
            "CHAMELEON",
            "DOUBLE_CHAMELEON",
            "TRIPLE_CHAMELEON",
        ]
        lightModel = LightModel(light_model_list=model_list)
        assert len(lightModel.profile_type_list) == len(model_list)

    def test_surface_brightness(self):
        output = self.LightModel.surface_brightness(
            x=1.0, y=1.0, kwargs_list=self.kwargs
        )
        npt.assert_almost_equal(output, 2.647127, decimal=6)

    def test_surface_brightness_array(self):
        output = self.LightModel.surface_brightness(
            x=[1], y=[1], kwargs_list=self.kwargs
        )
        npt.assert_almost_equal(output[0], 2.647127113888489, decimal=6)

    def test_functions_split(self):
        output = self.LightModel.functions_split(x=1.0, y=1.0, kwargs_list=self.kwargs)
        npt.assert_almost_equal(output[0][0], 0.058549831524319168, decimal=6)

    def test_param_name_list(self):
        param_name_list = self.LightModel.param_name_list
        assert len(self.light_model_list) == len(param_name_list)

    def test_param_name_list_latex(self):
        param_name_list = self.LightModel.param_name_list_latex
        assert len(self.light_model_list) == len(param_name_list)

    def test_num_param_linear(self):
        num = self.LightModel.num_param_linear(self.kwargs, list_return=False)
        assert num == 20

        num_list = self.LightModel.num_param_linear(self.kwargs, list_return=True)
        assert num_list[0] == 1

    def test_update_linear(self):
        response, n = self.LightModel.functions_split(1, 1, self.kwargs)
        param = np.ones(n) * 2
        kwargs_out, i = self.LightModel.update_linear(
            param, i=0, kwargs_list=self.kwargs
        )
        assert i == n
        assert kwargs_out[0]["amp"] == 2

    def test_total_flux(self):
        light_model_list = [
            "SERSIC",
            "SERSIC_ELLIPSE",
            "INTERPOL",
            "GAUSSIAN",
            "GAUSSIAN_ELLIPSE",
            "MULTI_GAUSSIAN",
            "MULTI_GAUSSIAN_ELLIPSE",
            "LINE_PROFILE",
        ]
        kwargs_list = [
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
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'SERSIC_ELLIPSE'
            {
                "image": np.ones((20, 5)),
                "scale": 1,
                "phi_G": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'INTERPOL'
            {"amp": 2, "sigma": 2, "center_x": 0, "center_y": 0},  # 'GAUSSIAN'
            {
                "amp": 2,
                "sigma": 2,
                "e1": 0.1,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'GAUSSIAN_ELLIPSE'
            {
                "amp": [1, 1],
                "sigma": [2, 1],
                "center_x": 0,
                "center_y": 0,
            },  # 'MULTI_GAUSSIAN'
            {
                "amp": [1, 1],
                "sigma": [2, 1],
                "e1": 0.1,
                "e2": 0,
                "center_x": 0,
                "center_y": 0,
            },  # 'MULTI_GAUSSIAN_ELLIPSE'
            {
                "amp": 1,
                "length": 1.0,
                "width": 0.01,
                "angle": 57,
                "start_x": 0,
                "start_y": 0,
            },  # 'LINE_PROFILE'
        ]
        lightModel = LightModel(light_model_list=light_model_list)
        total_flux_list = lightModel.total_flux(kwargs_list)
        assert total_flux_list[2] == 100
        assert total_flux_list[3] == 2
        assert total_flux_list[4] == 2
        assert total_flux_list[5] == 2
        assert total_flux_list[6] == 2

        total_flux_list = lightModel.total_flux(kwargs_list, norm=True)
        assert total_flux_list[2] == 100
        assert total_flux_list[3] == 1
        assert total_flux_list[4] == 1
        assert total_flux_list[5] == 2
        assert total_flux_list[6] == 2

    def test_delete_interpol_caches(self):
        x, y = util.make_grid(numPix=20, deltapix=1.0)
        gauss = Gaussian()
        flux = gauss.function(x, y, amp=1.0, center_x=0.0, center_y=0.0, sigma=1.0)
        image = util.array2image(flux)

        light_model_list = ["INTERPOL", "INTERPOL"]
        kwargs_list = [
            {"image": image, "scale": 1, "phi_G": 0, "center_x": 0, "center_y": 0},
            {"image": image, "scale": 1, "phi_G": 0, "center_x": 0, "center_y": 0},
        ]
        lightModel = LightModel(light_model_list=light_model_list)
        output = lightModel.surface_brightness(x, y, kwargs_list)
        for func in lightModel.func_list:
            assert hasattr(func, "_image_interp")
        lightModel.delete_interpol_caches()
        for func in lightModel.func_list:
            assert not hasattr(func, "_image_interp")

    def test_check_positive_flux_profile(self):
        ligthModel = LightModel(light_model_list=["GAUSSIAN"])
        kwargs_list = [{"amp": 0, "sigma": 1}]
        bool = ligthModel.check_positive_flux_profile(kwargs_list)
        assert bool

        kwargs_list = [{"amp": -1, "sigma": 1}]
        bool = ligthModel.check_positive_flux_profile(kwargs_list)
        assert not bool


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["WRONG"])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["UNIFORM"])
            lighModel.light_3d(r=1, kwargs_list=[{"amp": 1}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["UNIFORM"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.functions_split(x=0, y=0, kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["UNIFORM"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.num_param_linear(kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["UNIFORM"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.update_linear(param=[1], i=0, kwargs_list=[{}])
        with self.assertRaises(ValueError):
            lighModel = LightModel(light_model_list=["UNIFORM"])
            lighModel.profile_type_list = ["WRONG"]
            lighModel.total_flux(kwargs_list=[{}])


if __name__ == "__main__":
    pytest.main()

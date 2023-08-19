"""Tests for `galkin` module."""
import pytest
import numpy.testing as npt
import numpy as np
from lenstronomy.GalKin.light_profile import LightProfile
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.GalKin import velocity_util
import scipy.integrate as integrate


class TestLightProfile(object):
    def setup_method(self):
        pass

    def test_draw_light(self):
        np.random.seed(41)
        lightProfile = LightProfile(profile_list=["HERNQUIST"])
        kwargs_profile = [{"amp": 1.0, "Rs": 0.8}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=500000)
        bins = np.linspace(0.0, 1, 20)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        light2d = lightProfile.light_2d(
            R=(bins_hist[1:] + bins_hist[:-1]) / 2.0, kwargs_list=kwargs_profile
        )
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print(light2d / hist)
        for i in range(len(hist)):
            print(bins_hist[i], i, light2d[i] / hist[i])
            npt.assert_almost_equal(light2d[i] / hist[i], 1, decimal=1)

    def test_draw_light_2d_linear(self):
        np.random.seed(41)
        lightProfile = LightProfile(
            profile_list=["HERNQUIST"],
            interpol_grid_num=1000,
            max_interpolate=10,
            min_interpolate=0.01,
        )
        kwargs_profile = [{"amp": 1.0, "Rs": 0.8}]
        r_list = lightProfile.draw_light_2d_linear(kwargs_profile, n=100000)
        bins = np.linspace(0.0, 1, 20)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        light2d = lightProfile.light_2d(
            R=(bins_hist[1:] + bins_hist[:-1]) / 2.0, kwargs_list=kwargs_profile
        )
        light2d_upper = (
            lightProfile.light_2d(R=bins_hist[1:], kwargs_list=kwargs_profile)
            * bins_hist[1:]
        )
        light2d_lower = (
            lightProfile.light_2d(R=bins_hist[:-1], kwargs_list=kwargs_profile)
            * bins_hist[:-1]
        )
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.0
        print((light2d_upper - light2d_lower) / (light2d_upper + light2d_lower) * 2)
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print(light2d / hist)
        for i in range(2, len(hist)):
            print(bins_hist[i], i, light2d[i] / hist[i])
            npt.assert_almost_equal(light2d[i] / hist[i], 1, decimal=1)

    def test_draw_light_PJaffe(self):
        np.random.seed(41)
        lightProfile = LightProfile(profile_list=["PJAFFE"])
        kwargs_profile = [{"amp": 1.0, "Rs": 0.5, "Ra": 0.2}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=100000)
        bins = np.linspace(0, 2, 10)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        light2d = lightProfile.light_2d(
            R=(bins_hist[1:] + bins_hist[:-1]) / 2.0, kwargs_list=kwargs_profile
        )
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print(light2d / hist)
        npt.assert_almost_equal(light2d[8] / hist[8], 1, decimal=1)

        lightProfile = LightProfile(
            profile_list=["PJAFFE"], min_interpolate=0.0001, max_interpolate=20.0
        )
        kwargs_profile = [{"amp": 1.0, "Rs": 0.04, "Ra": 0.02}]
        r_list = lightProfile.draw_light_2d(kwargs_profile, n=100000)
        bins = np.linspace(0.0, 0.1, 10)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        light2d = lightProfile.light_2d(
            R=(bins_hist[1:] + bins_hist[:-1]) / 2.0, kwargs_list=kwargs_profile
        )
        light2d *= (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        print(light2d / hist)
        npt.assert_almost_equal(light2d[5] / hist[5], 1, decimal=1)
        assert hasattr(lightProfile, "_kwargs_light_circularized")
        lightProfile.delete_cache()
        if hasattr(lightProfile, "_kwargs_light_circularized"):
            assert False

    def test_draw_light_3d_hernquist(self):
        lightProfile = LightProfile(
            profile_list=["HERNQUIST"], min_interpolate=0.0001, max_interpolate=1000.0
        )
        kwargs_profile = [{"amp": 1.0, "Rs": 0.5}]
        r_list = lightProfile.draw_light_3d(
            kwargs_profile, n=1000000, new_compute=False
        )
        print(r_list, "r_list")
        # project it

        # test with draw light 2d profile routine
        # compare with 3d analytical solution vs histogram binned
        bins = np.linspace(0.0, 10, 20)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        bins_plot = (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light3d = lightProfile.light_3d(r=bins_plot, kwargs_list=kwargs_profile)
        light3d *= bins_plot**2
        light3d /= np.sum(light3d)
        hist /= np.sum(hist)
        # import matplotlib.pyplot as plt
        # plt.plot(bins_plot , light3d/light3d[5], label='3d reference Hernquist')
        # plt.plot(bins_plot, hist / hist[5], label='hist')
        # plt.legend()
        # plt.show()
        print(light3d / hist)
        # npt.assert_almost_equal(light3d / hist, 1, decimal=1)

        # compare with 2d analytical solution vs histogram binned
        # bins = np.linspace(0.1, 1, 10)
        R, x, y = velocity_util.project2d_random(np.array(r_list))
        hist_2d, bins_hist = np.histogram(R, bins=bins, density=True)
        hist_2d /= np.sum(hist_2d)
        bins_plot = (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light2d = lightProfile.light_2d(R=bins_plot, kwargs_list=kwargs_profile)
        light2d *= bins_plot**1
        light2d /= np.sum(light2d)

        light2d_finite = lightProfile.light_2d_finite(
            R=bins_plot, kwargs_list=kwargs_profile
        )
        light2d_finite *= bins_plot**1
        light2d_finite /= np.sum(light2d_finite)
        hist /= np.sum(hist)
        # import matplotlib.pyplot as plt
        # plt.plot(bins_plot, light2d/light2d[5], '--', label='2d reference Hernquist')
        # plt.plot(bins_plot, light2d_finite / light2d_finite[5], '-.', label='2d reference Hernquist finite')
        # plt.plot(bins_plot, hist_2d / hist_2d[5], label='hist')
        # plt.legend()
        # plt.show()
        print(light2d / hist_2d)

        # plt.plot(R, r_list, '.', label='test')
        # plt.legend()
        # plt.xlim([0, 0.2])
        # plt.ylim([0, 0.2])
        # plt.show()

        npt.assert_almost_equal(light2d / hist_2d, 1, decimal=1)

    def test_draw_light_3d_power_law(self):
        lightProfile = LightProfile(
            profile_list=["POWER_LAW"], min_interpolate=0.0001, max_interpolate=1000.0
        )
        kwargs_profile = [{"amp": 1.0, "gamma": 2, "e1": 0, "e2": 0}]
        r_list = lightProfile.draw_light_3d(
            kwargs_profile, n=1000000, new_compute=False
        )
        print(r_list, "r_list")
        # project it
        R, x, y = velocity_util.project2d_random(r_list)
        # test with draw light 2d profile routine

        # compare with 3d analytical solution vs histogram binned
        bins = np.linspace(0.1, 10, 10)
        hist, bins_hist = np.histogram(r_list, bins=bins, density=True)
        bins_plot = (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light3d = lightProfile.light_3d(r=bins_plot, kwargs_list=kwargs_profile)
        light3d *= bins_plot**2
        light3d /= np.sum(light3d)
        hist /= np.sum(hist)
        # import matplotlib.pyplot as plt
        # plt.plot(bins_plot , light3d/light3d[5], label='3d reference power-law')
        # plt.plot(bins_plot, hist / hist[5], label='hist')
        # plt.legend()
        # plt.show()
        print(light3d / hist)
        npt.assert_almost_equal(light3d / hist, 1, decimal=1)

        # compare with 2d analytical solution vs histogram binned
        # bins = np.linspace(0.1, 1, 10)
        hist, bins_hist = np.histogram(R, bins=bins, density=True)
        bins_plot = (bins_hist[1:] + bins_hist[:-1]) / 2.0
        light2d = lightProfile.light_2d_finite(R=bins_plot, kwargs_list=kwargs_profile)
        light2d *= bins_plot**1
        light2d /= np.sum(light2d)
        hist /= np.sum(hist)
        # import matplotlib.pyplot as plt
        # plt.plot(bins_plot , light2d/light2d[5], label='2d reference power-law')
        # plt.plot(bins_plot, hist / hist[5], label='hist')
        # plt.legend()
        # plt.show()
        print(light2d / hist)
        npt.assert_almost_equal(light2d / hist, 1, decimal=1)

    def test_ellipticity_in_profiles(self):
        np.random.seed(41)
        lightProfile = ["HERNQUIST_ELLIPSE", "PJAFFE_ELLIPSE"]
        import lenstronomy.Util.param_util as param_util

        phi, q = 0.14944144075912402, 0.4105628122365978
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = -0.37221683730659516, 0.70799587973181288
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)
        center_x = -0.019983826426838536
        center_y = 0.90000011282957304
        kwargs_profile = [
            {
                "Rs": 0.16350224766074103,
                "e1": e1,
                "e2": e2,
                "center_x": center_x,
                "center_y": center_y,
                "amp": 1.3168943578511678,
            },
            {
                "Rs": 0.29187068596715743,
                "e1": e12,
                "e2": e22,
                "center_x": center_x,
                "center_y": center_y,
                "Ra": 0.020000382843298824,
                "amp": 85.948773973262391,
            },
        ]
        kwargs_options = {
            "lens_model_list": ["SPEP"],
            "lens_light_model_list": lightProfile,
        }
        lensAnalysis = LightProfileAnalysis(LightModel(light_model_list=lightProfile))
        r_eff = lensAnalysis.half_light_radius(
            kwargs_profile,
            center_x=center_x,
            center_y=center_y,
            grid_spacing=0.1,
            grid_num=100,
        )
        kwargs_profile[0]["e1"], kwargs_profile[0]["e2"] = 0, 0
        kwargs_profile[1]["e1"], kwargs_profile[1]["e2"] = 0, 0
        r_eff_spherical = lensAnalysis.half_light_radius(
            kwargs_profile,
            center_x=center_x,
            center_y=center_y,
            grid_spacing=0.1,
            grid_num=100,
        )
        npt.assert_almost_equal(r_eff / r_eff_spherical, 1, decimal=2)

    def test_light_3d(self):
        np.random.seed(41)
        lightProfile = LightProfile(profile_list=["HERNQUIST"])
        r = np.logspace(-2, 2, 100)
        kwargs_profile = [{"amp": 1.0, "Rs": 0.5}]
        light_3d = lightProfile.light_3d_interp(r, kwargs_profile)
        light_3d_exact = lightProfile.light_3d(r, kwargs_profile)
        for i in range(len(r)):
            npt.assert_almost_equal(light_3d[i] / light_3d_exact[i], 1, decimal=3)

    def test_light_2d_finite(self):
        interpol_grid_num = 5000
        max_interpolate = 10
        min_interpolate = 0.0001
        lightProfile = LightProfile(
            profile_list=["HERNQUIST"],
            interpol_grid_num=interpol_grid_num,
            max_interpolate=max_interpolate,
            min_interpolate=min_interpolate,
        )
        kwargs_profile = [{"amp": 1.0, "Rs": 1.0}]

        # check whether projected light integral is the same as analytic expression
        R = 1.0

        I_R = lightProfile.light_2d_finite(R, kwargs_profile)
        out = integrate.quad(
            lambda x: lightProfile.light_3d(np.sqrt(R**2 + x**2), kwargs_profile),
            min_interpolate,
            np.sqrt(max_interpolate**2 - R**2),
        )
        l_R_quad = out[0] * 2

        npt.assert_almost_equal(l_R_quad / I_R, 1, decimal=2)

        l_R = lightProfile.light_2d(R, kwargs_profile)
        npt.assert_almost_equal(l_R / I_R, 1, decimal=2)

    def test_del_cache(self):
        lightProfile = LightProfile(profile_list=["HERNQUIST"])
        lightProfile._light_cdf = 1
        lightProfile._light_cdf_log = 2
        lightProfile._f_light_3d = 3
        lightProfile.delete_cache()
        assert hasattr(lightProfile, "_light_cdf") is False
        assert hasattr(lightProfile, "_light_cdf_log") is False
        assert hasattr(lightProfile, "_f_light_3d") is False


if __name__ == "__main__":
    pytest.main()

import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian, MultiGaussianEllipse
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.util as util


class TestLightAnalysis(object):
    def setup_method(self):
        pass

    def test_ellipticity(self):
        # GAUSSIAN
        e1_in = 0.1
        e2_in = 0
        kwargs_light = [
            {
                "amp": 1,
                "sigma": 1.0,
                "center_x": 0,
                "center_y": 0,
                "e1": e1_in,
                "e2": e2_in,
            }
        ]
        light_model_list = ["GAUSSIAN_ELLIPSE"]
        lensAnalysis = LightProfileAnalysis(
            LightModel(light_model_list=light_model_list)
        )
        e1, e2 = lensAnalysis.ellipticity(
            kwargs_light,
            center_x=0,
            center_y=0,
            model_bool_list=None,
            grid_spacing=0.1,
            grid_num=200,
        )
        npt.assert_almost_equal(e1, e1_in, decimal=4)
        npt.assert_almost_equal(e2, e2_in, decimal=4)

        # off-centered
        e1_in = 0.1
        e2_in = 0
        kwargs_light = [
            {
                "amp": 1,
                "sigma": 1.0,
                "center_x": 0.2,
                "center_y": 0,
                "e1": e1_in,
                "e2": e2_in,
            }
        ]
        light_model_list = ["GAUSSIAN_ELLIPSE"]
        lensAnalysis = LightProfileAnalysis(
            LightModel(light_model_list=light_model_list)
        )
        e1, e2 = lensAnalysis.ellipticity(
            kwargs_light, model_bool_list=None, grid_spacing=0.1, grid_num=200
        )
        npt.assert_almost_equal(e1, e1_in, decimal=4)
        npt.assert_almost_equal(e2, e2_in, decimal=4)

        # SERSIC
        e1_in = 0.1
        e2_in = 0
        kwargs_light = [
            {
                "amp": 1,
                "n_sersic": 2.0,
                "R_sersic": 1,
                "center_x": 0,
                "center_y": 0,
                "e1": e1_in,
                "e2": e2_in,
            }
        ]
        light_model_list = ["SERSIC_ELLIPSE"]
        lensAnalysis = LightProfileAnalysis(
            LightModel(light_model_list=light_model_list)
        )
        e1, e2 = lensAnalysis.ellipticity(
            kwargs_light,
            center_x=0,
            center_y=0,
            model_bool_list=None,
            grid_spacing=0.2,
            grid_num=400,
            iterative=True,
            num_iterative=30,
        )
        print(e1, e2)
        npt.assert_almost_equal(e1, e1_in, decimal=3)
        npt.assert_almost_equal(e2, e2_in, decimal=3)

        # Power-law
        e1_in = 0.3
        e2_in = 0
        center_x, center_y = 0.0, 0
        kwargs_light = [
            {
                "gamma": 2.0,
                "amp": 1,
                "center_x": center_x,
                "center_y": center_y,
                "e1": e1_in,
                "e2": e2_in,
            }
        ]
        light_model_list = ["POWER_LAW"]
        lensAnalysis = LightProfileAnalysis(
            LightModel(light_model_list=light_model_list)
        )
        e1, e2 = lensAnalysis.ellipticity(
            kwargs_light,
            center_x=center_x,
            center_y=center_y,
            model_bool_list=None,
            grid_spacing=0.05,
            grid_num=401,
            iterative=True,
            num_iterative=30,
        )
        print(e1, e2)
        npt.assert_almost_equal(e1, e1_in, decimal=2)
        npt.assert_almost_equal(e2, e2_in, decimal=3)

    def test_half_light_radius(self):
        Rs = 1.0
        kwargs_profile = [{"Rs": Rs, "amp": 1.0, "center_x": 0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lensAnalysis = LightProfileAnalysis(LightModel(**kwargs_options))
        r_eff_true = Rs / 0.551
        r_eff = lensAnalysis.half_light_radius(
            kwargs_profile, grid_num=500, grid_spacing=0.2
        )
        npt.assert_almost_equal(r_eff / r_eff_true, 1, 2)

        # now we shift the center
        Rs = 1.0
        kwargs_profile = [{"Rs": Rs, "amp": 1.0, "center_x": 1.0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lensAnalysis = LightProfileAnalysis(LightModel(**kwargs_options))
        r_eff_true = Rs / 0.551
        r_eff = lensAnalysis.half_light_radius(
            kwargs_profile, grid_num=500, grid_spacing=0.2
        )
        npt.assert_almost_equal(r_eff / r_eff_true, 1, 2)

        # now we add ellipticity
        Rs = 1.0
        kwargs_profile = [
            {
                "Rs": Rs,
                "amp": 1.0,
                "e1": 0.1,
                "e2": -0.1,
                "center_x": 0.0,
                "center_y": 0,
            }
        ]
        kwargs_options = {"light_model_list": ["HERNQUIST_ELLIPSE"]}
        lensAnalysis = LightProfileAnalysis(LightModel(**kwargs_options))
        r_eff_true = Rs / 0.551
        r_eff = lensAnalysis.half_light_radius(
            kwargs_profile, grid_num=500, grid_spacing=0.2
        )
        npt.assert_almost_equal(r_eff / r_eff_true, 1, 2)

    def test_radial_profile(self):
        Rs = 1.0
        kwargs_light = [{"Rs": Rs, "amp": 1.0, "center_x": 0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)
        r_list = np.linspace(start=0.01, stop=10, num=10)
        I_r = profile.radial_light_profile(
            r_list, kwargs_light, center_x=None, center_y=None, model_bool_list=None
        )
        I_r_true = lightModel.surface_brightness(r_list, 0, kwargs_light)
        npt.assert_almost_equal(I_r, I_r_true, decimal=5)

        # test off-center
        Rs = 1.0
        kwargs_light = [{"Rs": Rs, "amp": 1.0, "center_x": 1.0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)
        r_list = np.linspace(start=0.01, stop=10, num=10)
        I_r = profile.radial_light_profile(
            r_list, kwargs_light, center_x=None, center_y=None, model_bool_list=None
        )
        I_r_true = lightModel.surface_brightness(r_list + 1, 0, kwargs_light)
        npt.assert_almost_equal(I_r, I_r_true, decimal=5)

    def test_multi_gaussian_decomposition(self):
        Rs = 1.0
        kwargs_light = [{"Rs": Rs, "amp": 1.0, "center_x": 0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)

        amplitudes, sigmas, center_x, center_y = profile.multi_gaussian_decomposition(
            kwargs_light,
            grid_spacing=0.01,
            grid_num=100,
            model_bool_list=None,
            n_comp=20,
            center_x=None,
            center_y=None,
        )
        mge = MultiGaussian()
        r_array = np.logspace(start=-2, stop=0.5, num=10)
        print(r_array, "test r_array")
        flux = mge.function(
            r_array,
            0,
            amp=amplitudes,
            sigma=sigmas,
            center_x=center_x,
            center_y=center_y,
        )
        flux_true = lightModel.surface_brightness(r_array, 0, kwargs_light)
        npt.assert_almost_equal(flux / flux_true, 1, decimal=2)

        # test off-center

        Rs = 1.0
        offset = 1.0
        kwargs_light = [{"Rs": Rs, "amp": 1.0, "center_x": offset, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)

        amplitudes, sigmas, center_x, center_y = profile.multi_gaussian_decomposition(
            kwargs_light,
            grid_spacing=0.01,
            grid_num=100,
            model_bool_list=None,
            n_comp=20,
            center_x=None,
            center_y=None,
        )
        assert center_x == offset
        assert center_y == 0
        mge = MultiGaussian()
        r_array = np.logspace(start=-2, stop=0.5, num=10)
        print(r_array, "test r_array")
        flux = mge.function(
            r_array,
            0,
            amp=amplitudes,
            sigma=sigmas,
            center_x=center_x,
            center_y=center_y,
        )
        flux_true = lightModel.surface_brightness(r_array, 0, kwargs_light)
        npt.assert_almost_equal(flux / flux_true, 1, decimal=2)

        # import matplotlib.pyplot as plt
        # plt.loglog(r_array, flux, label='mge')
        # plt.loglog(r_array, flux_true, label='true')
        # plt.legend()
        # plt.show()

    def test_multi_gaussian_decomposition_ellipse(self):
        Rs = 1.0
        kwargs_light = [{"Rs": Rs, "amp": 1.0, "center_x": 0, "center_y": 0}]
        kwargs_options = {"light_model_list": ["HERNQUIST"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)

        kwargs_mge = profile.multi_gaussian_decomposition_ellipse(
            kwargs_light,
            grid_spacing=0.01,
            grid_num=100,
            model_bool_list=None,
            n_comp=20,
            center_x=None,
            center_y=None,
        )
        mge = MultiGaussianEllipse()
        r_array = np.logspace(start=-2, stop=0.5, num=10)
        flux = mge.function(r_array, 0, **kwargs_mge)
        flux_true = lightModel.surface_brightness(r_array, 0, kwargs_light)
        npt.assert_almost_equal(flux / flux_true, 1, decimal=2)

        # elliptic

        Rs = 1.0
        kwargs_light = [
            {"Rs": Rs, "amp": 1.0, "e1": 0.1, "e2": 0, "center_x": 0, "center_y": 0}
        ]
        kwargs_options = {"light_model_list": ["HERNQUIST_ELLIPSE"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)

        kwargs_mge = profile.multi_gaussian_decomposition_ellipse(
            kwargs_light,
            grid_spacing=0.1,
            grid_num=400,
            model_bool_list=None,
            n_comp=20,
            center_x=None,
            center_y=None,
        )

        print(kwargs_mge["e1"])
        mge = MultiGaussianEllipse()
        r_array = np.logspace(start=-2, stop=0.5, num=10)
        flux = mge.function(r_array, 0, **kwargs_mge)
        flux_true = lightModel.surface_brightness(r_array, 0, kwargs_light)

        npt.assert_almost_equal(flux / flux_true, 1, decimal=1)

    def test_flux_components(self):
        amp = 1
        kwargs_profile = [{"amp": amp}]
        kwargs_options = {"light_model_list": ["UNIFORM"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)
        grid_num = 40
        grid_spacing = 0.1
        flux_list, R_h_list = profile.flux_components(
            kwargs_profile, grid_num=grid_num, grid_spacing=grid_spacing
        )
        assert len(flux_list) == 1
        area = (grid_num * grid_spacing) ** 2
        npt.assert_almost_equal(flux_list[0], area * amp, decimal=8)

        phi, q = -0.37221683730659516, 0.70799587973181288
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)

        phi2, q2 = 0.14944144075912402, 0.4105628122365978
        e12, e22 = param_util.phi_q2_ellipticity(phi2, q2)

        kwargs_profile = [
            {
                "Rs": 0.16350224766074103,
                "e1": e12,
                "e2": e22,
                "center_x": -0.019983826426838536,
                "center_y": 0.90000011282957304,
                "amp": 1.3168943578511678,
            },
            {
                "Rs": 0.29187068596715743,
                "e1": e1,
                "e2": e2,
                "center_x": 0.020568531548241405,
                "center_y": 0.036038490364800925,
                "Ra": 0.020000382843298824,
                "amp": 85.948773973262391,
            },
        ]
        kwargs_options = {"light_model_list": ["HERNQUIST_ELLIPSE", "PJAFFE_ELLIPSE"]}
        lightModel = LightModel(**kwargs_options)
        profile = LightProfileAnalysis(light_model=lightModel)

        flux_list, R_h_list = profile.flux_components(
            kwargs_profile, grid_num=400, grid_spacing=0.01
        )
        assert len(flux_list) == 2
        npt.assert_almost_equal(flux_list[0], 0.1940428118053717, decimal=6)
        npt.assert_almost_equal(flux_list[1], 3.0964046927612707, decimal=6)


"""

    def test_light2mass_mge(self):
        from lenstronomy.LightModel.Profiles.gaussian import MultiGaussianEllipse
        multiGaussianEllipse = MultiGaussianEllipse()
        x_grid, y_grid = util.make_grid(numPix=100, deltapix=0.05)
        kwargs_light = [{'amp': [2, 1], 'sigma': [0.1, 1], 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0}]
        light_model_list = ['MULTI_GAUSSIAN_ELLIPSE']
        lensAnalysis = ProfileAnalysis(kwargs_model={'lens_light_model_list': light_model_list})
        kwargs_mge = lensAnalysis.light2mass_mge(kwargs_lens_light=kwargs_light, numPix=100, deltaPix=0.05, elliptical=True)
        npt.assert_almost_equal(kwargs_mge['e1'], kwargs_light[0]['e1'], decimal=2)

        del kwargs_light[0]['center_x']
        del kwargs_light[0]['center_y']
        kwargs_mge = lensAnalysis.light2mass_mge(kwargs_lens_light=kwargs_light, numPix=100, deltaPix=0.05,
                                                 elliptical=False)
        npt.assert_almost_equal(kwargs_mge['center_x'], 0, decimal=2)

    def test_light2mass_mge_elliptical_sersic(self):
        # same test as above but with Sersic ellipticity definition
        lens_light_kwargs = [
            {'R_sersic': 1.3479852771734446, 'center_x': -0.0014089381116285044, 'n_sersic': 2.260502794737016,
             'amp': 0.08679965264978318, 'center_y': 0.0573684892835563, 'e1': 0.22781838418202335,
             'e2': 0.03841125245832406},
            {'R_sersic': 0.20907637464009315, 'center_x': -0.0014089381116285044, 'n_sersic': 3.0930684763455156,
             'amp': 3.2534559112899633, 'center_y': 0.0573684892835563, 'e1': 0.0323604434989261,
             'e2': -0.12430547471424626}]
        light_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
        lensAnalysis = ProfileAnalysis({'lens_light_model_list': light_model_list})
        kwargs_mge = lensAnalysis.light2mass_mge(lens_light_kwargs, model_bool_list=None, elliptical=True, numPix=500,
                                                 deltaPix=0.5)
        print(kwargs_mge)
        npt.assert_almost_equal(kwargs_mge['e1'], 0.22, decimal=2)

    def test_mass_fraction_within_radius(self):
        center_x, center_y = 0.5, -1
        theta_E = 1.1
        kwargs_lens = [{'theta_E': 1.1, 'center_x': center_x, 'center_y': center_y}]
        lensAnalysis = ProfileAnalysis(kwargs_model={'lens_model_list': ['SIS']})
        kappa_mean_list = lensAnalysis.mass_fraction_within_radius(kwargs_lens, center_x, center_y, theta_E, numPix=100)
        npt.assert_almost_equal(kappa_mean_list[0], 1, 2)

    def test_point_source(self):
        kwargs_model = {'lens_model_list': ['SPEMD', 'SHEAR_GAMMA_PSI'], 'point_source_model_list': ['SOURCE_POSITION']}
        lensAnalysis = ProfileAnalysis(kwargs_model=kwargs_model)
        source_x, source_y = 0.02, 0.1
        kwargs_ps = [{'dec_source': source_y, 'ra_source': source_x, 'point_amp': 75.155}]
        kwargs_lens = [{'e2': 0.1, 'center_x': 0, 'theta_E': 1.133, 'e1': 0.1, 'gamma': 2.063, 'center_y': 0}, {'gamma_ext': 0.026, 'psi_ext': 1.793}]
        x_image, y_image = lensAnalysis.PointSource.image_position(kwargs_ps=kwargs_ps, kwargs_lens=kwargs_lens)
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
        from lenstronomy.LensModel.lens_model import LensModel
        lens_model = LensModel(lens_model_list=['SPEMD', 'SHEAR_GAMMA_PSI'])
        from lenstronomy.PointSource.point_source import PointSource
        ps = PointSource(point_source_type_list=['SOURCE_POSITION'], lens_model=lens_model)
        x_image_new, y_image_new = ps.image_position(kwargs_ps, kwargs_lens)
        npt.assert_almost_equal(x_image_new[0], x_image[0], decimal=7)

        solver = LensEquationSolver(lens_model=lens_model)

        x_image_true, y_image_true = solver.image_position_from_source(source_x, source_y, kwargs_lens, min_distance=0.01, search_window=5,
                                   precision_limit=10**(-10), num_iter_max=100, arrival_time_sort=True,
                                   initial_guess_cut=False, verbose=False, x_center=0, y_center=0, num_random=0,
                                   non_linear=False, magnification_limit=None)

        print(x_image[0], y_image[0], x_image_true, y_image_true)
        npt.assert_almost_equal(x_image_true, x_image[0], decimal=7)

    def test_lens_center(self):
        center_x, center_y = 0.43, -0.67
        kwargs_lens = [{'theta_E': 1, 'center_x': center_x, 'center_y': center_y}]
        profileAnalysis = ProfileAnalysis({'lens_model_list': ['SIS']})
        center_x_out, center_y_out = profileAnalysis.lens_center(kwargs_lens)
        npt.assert_almost_equal(center_x_out, center_x, 2)
        npt.assert_almost_equal(center_y_out, center_y, 2)



            analysis = ProfileAnalysis(kwargs_model={'lens_model_list': ['SIS']})
            analysis.multi_gaussian_lens(kwargs_lens=[{'theta_E'}])
        with self.assertRaises(ValueError):
            analysis = ProfileAnalysis(kwargs_model={'lens_light_model_list': ['GAUSSIAN']})
            analysis.flux_components(kwargs_light=[{}], n_grid=400, delta_grid=0.01, deltaPix=1., type="wrong")

"""


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            raise ValueError()


if __name__ == "__main__":
    pytest.main()

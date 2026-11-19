__author__ = "sibirrer"

import copy

import pytest
import numpy.testing as npt
import numpy as np
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Sampling.sampler import Sampler
from lenstronomy.Sampling.Samplers.dynesty_sampler import DynestySampler


class TestFittingSequence(object):
    """Test the fitting sequences."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        num_pix = 10  # cutout pixel size
        delta_pix = 0.05  # pixel size in arcsec (area per pixel = delta_pix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        self.kwargs_data = sim_util.data_configure_simple(
            num_pix, delta_pix, exp_time, sigma_bkg
        )
        data_class = ImageData(**self.kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": delta_pix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)
        self.kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": psf_gaussian.kernel_point_source,
            "psf_variance_map": np.zeros_like(psf_gaussian.kernel_point_source),
        }
        psf_class = PSF(**self.kwargs_psf)
        # 'EXTERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angle (in radian)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_model_list = ["SPEP", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [
            {"ra_source": 0.0, "dec_source": 0.0, "source_amp": 1.0}
        ]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = ["SOURCE_POSITION"]
        point_source_class = PointSource(
            point_source_type_list=point_source_list, fixed_magnification_list=[True]
        )
        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
            "compute_mode": "regular",
            "point_source_supersampling_factor": 1,
        }
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )

        data_class.update_data(image_sim)
        self.data_class = data_class
        self.psf_class = psf_class
        self.kwargs_data["image_data"] = image_sim
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": point_source_list,
            "fixed_magnification_list": [False],
            "index_lens_model_list": [[0, 1]],
            "point_source_frame_list": [[0]],
        }
        self.kwargs_numerics = kwargs_numerics

        num_source_model = len(source_model_list)

        self.kwargs_constraints = {
            "num_point_source_list": [4],
            "image_plane_source_list": [False] * num_source_model,
            "solver_type": "NONE",  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
        }

        self.kwargs_likelihood = {
            "force_no_add_image": True,
            "source_marg": True,
            "linear_prior": [1],
            "image_position_uncertainty": 0.004,
            "source_position_tolerance": None,
            "source_position_sigma": 0.001,
            "check_positive_flux": True,
        }

        lens_sigma = [
            {
                "theta_E": 0.1,
                "gamma": 0.1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.1,
                "center_y": 0.1,
            },
            {"gamma1": 0.1, "gamma2": 0.1},
        ]
        lens_lower = [
            {
                "theta_E": 0.0,
                "gamma": 1.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            },
            {"gamma1": -0.3, "gamma2": -0.3},
        ]
        lens_upper = [
            {
                "theta_E": 10.0,
                "gamma": 2.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            },
            {"gamma1": 0.3, "gamma2": 0.3},
        ]
        source_sigma = [
            {
                "R_sersic": 0.05,
                "n_sersic": 0.5,
                "center_x": 0.1,
                "center_y": 0.1,
                "e1": 0.1,
                "e2": 0.1,
            }
        ]
        source_lower = [
            {
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        source_upper = [
            {
                "R_sersic": 10,
                "n_sersic": 5.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        lens_light_sigma = [
            {"R_sersic": 0.05, "n_sersic": 0.5, "center_x": 0.1, "center_y": 0.1}
        ]
        lens_light_lower = [
            {"R_sersic": 0.01, "n_sersic": 0.5, "center_x": -2, "center_y": -2}
        ]
        lens_light_upper = [
            {"R_sersic": 10, "n_sersic": 5.5, "center_x": 2, "center_y": 2}
        ]
        ps_sigma = [{"ra_source": 1, "dec_source": 1, "point_amp": 1}]

        lens_param = (
            self.kwargs_lens,
            lens_sigma,
            [{}, {"ra_0": 0, "dec_0": 0}],
            lens_lower,
            lens_upper,
        )
        source_param = (
            self.kwargs_source,
            source_sigma,
            [{}],
            source_lower,
            source_upper,
        )
        lens_light_param = (
            self.kwargs_lens_light,
            lens_light_sigma,
            [{"center_x": 0}],
            lens_light_lower,
            lens_light_upper,
        )
        ps_param = self.kwargs_ps, ps_sigma, [{}], self.kwargs_ps, self.kwargs_ps

        self.kwargs_params = {
            "lens_model": lens_param,
            "source_model": source_param,
            "lens_light_model": lens_light_param,
            "point_source_model": ps_param,
            # 'special': special_param
        }
        image_band = [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        multi_band_list = [image_band]
        self.kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "multi-linear",
        }

    def test_simulationAPI_image(self):
        npt.assert_almost_equal(self.data_class.data[4, 4], 0.1, decimal=0)

    def test_simulationAPI_psf(self):
        npt.assert_almost_equal(
            np.sum(self.psf_class.kernel_point_source), 1, decimal=6
        )

    # Tests for validating the initial parameter spreads (kwargs_sigma) and the
    # sigma_scale argument before sampling starts. Invalid settings must raise before
    # the sampler is constructed. The test_*_sigma_handover tests replace only the
    # sampling step with a fake that records its input, to check the exact spreads that
    # FittingSequence hands to each sampler. All other tests use the real samplers.

    def _fitting_sequence(self):
        return FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )

    @staticmethod
    def _run_sampler(fitting, method, sigma_scale=1):
        """Start one sampler with the minimum settings needed to get it going."""
        if method == "pso":
            fitting.pso(n_particles=4, n_iterations=1, sigma_scale=sigma_scale)
        elif method in ["emcee", "zeus"]:
            fitting.mcmc(
                n_burn=0,
                n_run=1,
                walkerRatio=2,
                sigma_scale=sigma_scale,
                sampler_type=method,
                progress=False,
            )
        elif method == "Cobaya":
            # Cobaya has no sigma_scale argument and uses kwargs_sigma unscaled
            fitting.fit_sequence([["Cobaya", {}]])
        elif method == "nested_gaussian":
            fitting.nested_sampling(
                sampler_type="dynesty", prior_type="gaussian", sigma_scale=sigma_scale
            )
        else:
            raise ValueError(method)

    @pytest.mark.parametrize(
        "method", ["pso", "emcee", "zeus", "Cobaya", "nested_gaussian"]
    )
    @pytest.mark.parametrize("sigma", [0, -0.1, np.nan, np.inf])
    def test_invalid_initial_sigma(self, method, sigma):
        """A zero, negative or non-finite spread on a free parameter (here the free lens
        parameter theta_E) must raise a ValueError that names that parameter.

        The error comes from FittingSequence._initial_sigma before the sampler is
        constructed. The message match ensures that the test does not pass on an
        unrelated error raised later inside a sampler.
        """
        self.kwargs_params["lens_model"][1][0]["theta_E"] = sigma
        fitting = self._fitting_sequence()
        with pytest.raises(
            ValueError, match="Initial sigma must be finite.*theta_E_lens0"
        ):
            self._run_sampler(fitting, method)

    @pytest.mark.parametrize("method", ["pso", "emcee", "zeus", "nested_gaussian"])
    @pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
    def test_invalid_sigma_scale(self, method, scale):
        """A sigma_scale that is not a finite positive number must raise a ValueError
        before sampling, even when every kwargs_sigma entry is valid."""
        fitting = self._fitting_sequence()
        with pytest.raises(ValueError, match="sigma_scale must be finite"):
            self._run_sampler(fitting, method, sigma_scale=scale)

    def test_initial_sigma_values(self):
        """_initial_sigma returns kwargs_sigma in the order of the free parameters,
        multiplied by sigma_scale, and leaves out fixed parameters."""
        fitting = self._fitting_sequence()
        param_class = fitting.param_class
        _, names = param_class.num_param()
        expected = np.array(
            param_class.kwargs2args(**fitting._updateManager.sigma_kwargs)
        )
        npt.assert_array_equal(fitting._initial_sigma(param_class), expected)
        npt.assert_array_equal(fitting._initial_sigma(param_class, 3), 3 * expected)
        # ra_0/dec_0 of the shear and center_x of the lens light are fixed
        assert len(expected) == len(names)
        assert "ra_0_lens1" not in names
        assert "center_x_lens_light0" not in names

    def test_fixed_parameter_zero_sigma(self):
        """A zero spread is allowed on a fixed parameter, because no sampler draws from
        it.

        A real (one-iteration) PSO run must succeed and keep the fixed value.
        """
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0
        self.kwargs_params["lens_model"][2][0]["theta_E"] = 1
        fitting = self._fitting_sequence()
        result, _, names = fitting.pso(n_particles=4, n_iterations=1)
        assert result["kwargs_lens"][0]["theta_E"] == 1
        assert "theta_E_lens0" not in names

    def test_reused_mcmc_samples_ignore_sigma(self):
        """When re_use_samples is True and init_samples is given, emcee starts its
        walkers from those samples and draws nothing from kwargs_sigma.

        An invalid spread or sigma_scale must therefore not raise in that case. This
        test runs real emcee for one step.
        """
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0
        # The fixture pins the point-source parameters (lower == upper). Widen them
        # here so that emcee accepts linearly independent walkers inside the bounds.
        ps_init, ps_sigma, ps_fixed, _, _ = self.kwargs_params["point_source_model"]
        ps_lower = [{k: v - 1 for k, v in kw.items()} for kw in ps_init]
        ps_upper = [{k: v + 1 for k, v in kw.items()} for kw in ps_init]
        self.kwargs_params["point_source_model"] = (
            ps_init,
            ps_sigma,
            ps_fixed,
            ps_lower,
            ps_upper,
        )
        fitting = self._fitting_sequence()
        param_class = fitting.param_class
        n_param, _ = param_class.num_param()
        start = np.array(
            param_class.kwargs2args(**fitting._updateManager.parameter_state)
        )
        lower = np.array(param_class.kwargs2args(**fitting._updateManager.lower_kwargs))
        upper = np.array(param_class.kwargs2args(**fitting._updateManager.upper_kwargs))
        # small independent offsets around the initial state, kept inside the bounds
        rng = np.random.default_rng(42)
        offsets = 1e-4 * rng.standard_normal((4 * n_param, n_param))
        init_samples = np.clip(start + offsets, lower, upper)
        n_walkers = 2 * n_param
        output = fitting.mcmc(
            n_burn=0,
            n_run=1,
            n_walkers=n_walkers,
            init_samples=init_samples,
            re_use_samples=True,
            sigma_scale=0,
            progress=False,
        )
        sampler_type, samples, param_list, _ = output
        assert sampler_type == "emcee"
        assert samples.shape == (n_walkers, n_param)
        assert np.all((samples >= lower) & (samples <= upper))
        assert len(param_list) == n_param

    def test_nested_sampling_priors(self):
        """With a uniform prior, nested sampling does not use kwargs_sigma, so an
        invalid spread must not raise.

        With a Gaussian prior, the spreads are returned unscaled, because the nested
        samplers apply sigma_scale themselves.
        """
        fitting = self._fitting_sequence()
        param_class = fitting.param_class
        expected = np.array(
            param_class.kwargs2args(**fitting._updateManager.sigma_kwargs)
        )
        _, sigma = fitting._prepare_sampling("gaussian", sigma_scale=5)
        npt.assert_array_equal(sigma, expected)

        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0
        fitting = self._fitting_sequence()
        assert fitting._prepare_sampling("uniform", sigma_scale=0) == (None, None)

    @pytest.mark.parametrize("method", ["emcee", "zeus"])
    def test_mcmc_sigma_handover(self, method, monkeypatch):
        """Fresh emcee and zeus walkers are drawn with kwargs_sigma * sigma_scale.

        With a theta_E spread of 0.2 and sigma_scale=3, the sampler must receive 0.6.
        """
        calls = []

        def _fake_mcmc(self, n_walkers, n_run, n_burn, mean_start, sigma_start, **kw):
            calls.append(sigma_start)
            return np.zeros((n_walkers, len(mean_start))), np.zeros(n_walkers)

        monkeypatch.setattr(Sampler, "mcmc_" + method, _fake_mcmc)
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0.2
        fitting = self._fitting_sequence()
        fitting.mcmc(
            n_burn=0, n_run=1, walkerRatio=2, sigma_scale=3, sampler_type=method
        )
        _, names = fitting.param_class.num_param()
        npt.assert_almost_equal(calls[0][names.index("theta_E_lens0")], 0.6, decimal=10)

    def test_pso_sigma_handover(self, monkeypatch):
        """PSO particles start within init +/- kwargs_sigma * sigma_scale.

        With theta_E = 1, a spread of 0.2 and sigma_scale=3, the sampler must receive
        the range [0.4, 1.6], which lies inside the theta_E limits [0, 10].
        """
        calls = []

        def _fake_pso(self, n_particles, n_iterations, lower, upper, init_pos, **kw):
            calls.append((lower, upper))
            return init_pos, []

        monkeypatch.setattr(Sampler, "pso", _fake_pso)
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0.2
        fitting = self._fitting_sequence()
        fitting.pso(n_particles=4, n_iterations=1, sigma_scale=3)
        _, names = fitting.param_class.num_param()
        lower, upper = calls[0]
        index = names.index("theta_E_lens0")
        npt.assert_almost_equal(lower[index], 0.4, decimal=10)
        npt.assert_almost_equal(upper[index], 1.6, decimal=10)

    def test_cobaya_sigma_handover(self, monkeypatch):
        """Cobaya receives kwargs_sigma unchanged, as it has no sigma_scale argument."""
        calls = []

        class _FakeCobayaSampler(object):
            def __init__(self, likelihood_class, mean_start, sigma_start):
                calls.append(sigma_start)
                self._mean_start = mean_start

            def run(self, **kwargs):
                return {}, "mcmc", self._mean_start

        monkeypatch.setattr(
            "lenstronomy.Workflow.fitting_sequence.CobayaSampler", _FakeCobayaSampler
        )
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0.2
        fitting = self._fitting_sequence()
        fitting.fit_sequence([["Cobaya", {}]])
        _, names = fitting.param_class.num_param()
        npt.assert_almost_equal(calls[0][names.index("theta_E_lens0")], 0.2, decimal=10)

    def test_nested_sampling_sigma_handover(self, monkeypatch):
        """The Gaussian prior width is kwargs_sigma * sigma_scale, with sigma_scale
        applied exactly once.

        The real DynestySampler is constructed, because it applies sigma_scale in its
        constructor. Only its run() is replaced. With a theta_E spread of 0.2 and
        sigma_scale=3, the prior width must be 0.6 (and not 1.8 from scaling twice).
        """
        calls = []

        def _fake_run(self, kwargs_run):
            calls.append(self.sigmas)
            return np.array([self.means]), self.means, 0, 0, np.zeros(1), {}

        monkeypatch.setattr(DynestySampler, "run", _fake_run)
        self.kwargs_params["lens_model"][1][0]["theta_E"] = 0.2
        fitting = self._fitting_sequence()
        fitting.nested_sampling(
            sampler_type="dynesty", prior_type="gaussian", sigma_scale=3
        )
        _, names = fitting.param_class.num_param()
        npt.assert_almost_equal(calls[0][names.index("theta_E_lens0")], 0.6, decimal=10)

    def test_fitting_sequence(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )

        kwargs_result = fittingSequence.best_fit(bijective=False)
        lens_temp = kwargs_result["kwargs_lens"]
        npt.assert_almost_equal(
            lens_temp[0]["theta_E"], self.kwargs_lens[0]["theta_E"], decimal=2
        )

        logL = fittingSequence.best_fit_likelihood()
        print(logL, "test")
        # print(lens_temp, source_temp, lens_light_temp, ps_temp, special_temp)
        assert logL < 0
        bic = fittingSequence.bic
        assert bic > 0
        # npt.assert_almost_equal(bic, 20000000220.29376, decimal=-4)

        # npt.assert_almost_equal(logL, -10000000061.792593, decimal=-4)

        n_p = 2
        n_i = 2
        fitting_list = []

        kwargs_pso = {"sigma_scale": 1, "n_particles": n_p, "n_iterations": n_i}
        fitting_list.append(["PSO", kwargs_pso])
        kwargs_align = {"delta_shift": 0.2, "n_particles": 2, "n_iterations": 2}
        fitting_list.append(["align_images", kwargs_align])
        kwargs_psf_iter = {
            "num_iter": 2,
            "psf_iter_factor": 0.5,
            "stacking_method": "mean",
            "new_procedure": False,
        }
        fitting_list.append(["psf_iteration", kwargs_psf_iter])
        fitting_list.append(["restart", None])
        fitting_list.append(["fix_not_computed", {"free_bands": [True]}])
        n_sersic_overwrite = 4
        kwargs_update = {
            "lens_light_add_fixed": [[0, ["n_sersic"], [n_sersic_overwrite]]],
            "lens_light_remove_fixed": [[0, ["center_x"]]],
            "change_source_lower_limit": [[0, ["n_sersic"], [0.1]]],
            "change_source_upper_limit": [[0, ["n_sersic"], [10]]],
        }
        fitting_list.append(["update_settings", kwargs_update])

        chain_list = fittingSequence.fit_sequence(fitting_list)
        (
            lens_fixed,
            source_fixed,
            lens_light_fixed,
            ps_fixed,
            special_fixed,
            extinction_fixed,
            tracer_source_fixed,
        ) = fittingSequence._updateManager.fixed_kwargs
        kwargs_result = fittingSequence.best_fit(bijective=False)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"],
            self.kwargs_lens[0]["theta_E"],
            decimal=1,
        )
        npt.assert_almost_equal(
            fittingSequence._updateManager._lens_light_fixed[0]["n_sersic"],
            n_sersic_overwrite,
            decimal=8,
        )
        npt.assert_almost_equal(lens_light_fixed[0]["n_sersic"], 4, decimal=-1)
        assert fittingSequence._updateManager._lower_kwargs[1][0]["n_sersic"] == 0.1
        assert fittingSequence._updateManager._upper_kwargs[1][0]["n_sersic"] == 10

        # test 'set_param_value' fitting sequence
        fitting_list = [
            ["set_param_value", {"lens": [[1, ["gamma1"], [0.013]]]}],
            ["set_param_value", {"lens_light": [[0, ["center_x"], [0.009]]]}],
            ["set_param_value", {"source": [[0, ["n_sersic"], [2.993]]]}],
            ["set_param_value", {"ps": [[0, ["ra_source"], [0.007]]]}],
        ]

        fittingSequence.fit_sequence(fitting_list)

        kwargs_set = fittingSequence._updateManager.parameter_state
        assert kwargs_set["kwargs_lens"][1]["gamma1"] == 0.013
        assert kwargs_set["kwargs_lens_light"][0]["center_x"] == 0.009
        assert kwargs_set["kwargs_source"][0]["n_sersic"] == 2.993
        assert kwargs_set["kwargs_ps"][0]["ra_source"] == 0.007

        # test `set_amplitudes` fitting sequence
        # amplitudes were initialized at 1.0
        assert kwargs_set["kwargs_source"][0]["amp"] == 1.0
        assert kwargs_set["kwargs_lens_light"][0]["amp"] == 1.0
        # Updates all amplitudes to values obtained from linear solver
        fitting_list = [["set_amplitudes", {}]]
        fittingSequence.fit_sequence(fitting_list)
        kwargs_set = fittingSequence._updateManager.parameter_state
        assert kwargs_set["kwargs_source"][0]["amp"] != 1.0
        assert kwargs_set["kwargs_lens_light"][0]["amp"] != 1.0

        from unittest import TestCase

        t = TestCase()
        with t.assertRaises(ValueError):
            fitting_list_two = []
            fitting_list_two.append(["fake_mcmc_method", kwargs_pso])
            fittingSequence.fit_sequence(fitting_list_two)

        with t.assertRaises(ValueError):
            # should raise a value error for n_walkers = walkerRatio = None
            fitting_list_three = []
            kwargs_test = {"n_burn": 10, "n_run": 10}
            fitting_list_three.append(["emcee", kwargs_test])
            fittingSequence.fit_sequence(fitting_list_three)

        psf_iteration_list = fittingSequence.psf_iteration_memory
        assert len(psf_iteration_list) == 1
        assert "sequence" in psf_iteration_list[0]
        assert "band" in psf_iteration_list[0]
        assert "psf_before" in psf_iteration_list[0]
        assert "psf_after" in psf_iteration_list[0]

    def test_cobaya(self):
        np.random.seed(42)

        # make a basic lens model to fit
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        num_pix = 10  # cutout pixel size
        delta_pix = 0.05  # pixel size in arcsec (area per pixel = delta_pix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification
        kwargs_data = sim_util.data_configure_simple(
            num_pix, delta_pix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": delta_pix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)

        # make a lens
        lens_model_list = ["SIS"]
        kwargs_lens = [{"theta_E": 1.5, "center_x": 0.0, "center_y": 0.0}]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # make a source
        source_model_list = ["SERSIC"]
        kwargs_source = [
            {
                "amp": 1.0,
                "R_sersic": 0.3,
                "n_sersic": 3.0,
                "center_x": 0.1,
                "center_y": 0.1,
            }
        ]
        source_model_class = LightModel(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            data_class,
            psf_gaussian,
            lens_model_class,
            source_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(imageModel, kwargs_lens, kwargs_source)

        data_class.update_data(image_sim)

        kwargs_data["image_data"] = image_sim

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
        }

        lens_fixed = [{"center_x": 0.0, "center_y": 0.0}]
        lens_sigma = [{"theta_E": 0.01, "center_x": 0.1, "center_y": 0.1}]
        lens_lower = [{"theta_E": 0.1, "center_x": -10, "center_y": -10}]
        lens_upper = [{"theta_E": 3.0, "center_x": 10, "center_y": 10}]

        source_fixed = [{}]
        source_sigma = [
            {"R_sersic": 0.01, "n_sersic": 0.01, "center_x": 0.01, "center_y": 0.01}
        ]
        source_lower = [
            {"R_sersic": 0.01, "n_sersic": 0.5, "center_x": -1, "center_y": -1}
        ]
        source_upper = [
            {"R_sersic": 1.0, "n_sersic": 6.0, "center_x": 1, "center_y": 1}
        ]

        lens_param = [kwargs_lens, lens_sigma, lens_fixed, lens_lower, lens_upper]
        source_param = [
            kwargs_source,
            source_sigma,
            source_fixed,
            source_lower,
            source_upper,
        ]

        kwargs_params = {"lens_model": lens_param, "source_model": source_param}

        kwargs_constraints = {}

        multi_band_list = [[kwargs_data, kwargs_psf_gaussian, kwargs_numerics]]

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "multi-linear",
        }

        kwargs_likelihood = {"source_marg": False}

        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
        )

        kwargs_cobaya = {
            "proposal_widths": [0.001, 0.001, 0.001, 0.001, 0.001],
            "Rminus1_stop": 100,  # does this need to be large? can we run in test mode?
            "force_overwrite": True,
        }

        chain_list = fittingSequence.fit_sequence([["Cobaya", kwargs_cobaya]])

    def test_zeus(self):
        np.random.seed(42)
        # we make a very basic lens+source model to feed to check zeus can be run through fitting sequence
        # we don't use the kwargs defined in setup() as those are modified during the tests; using unique kwargs here is safer

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        num_pix = 10  # cutout pixel size
        delta_pix = 0.05  # pixel size in arcsec (area per pixel = delta_pix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            num_pix, delta_pix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": delta_pix,
            "truncation": 3,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)

        # make a lens
        lens_model_list = ["EPL"]
        kwargs_epl = {
            "theta_E": 0.6,
            "gamma": 2.6,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": 0.1,
        }
        kwargs_lens = [kwargs_epl]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # make a source
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": 0.1,
            "e2": 0.1,
        }
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            data_class,
            psf_gaussian,
            lens_model_class,
            source_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(imageModel, kwargs_lens, kwargs_source)

        data_class.update_data(image_sim)

        kwargs_data["image_data"] = image_sim

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
        }

        lens_fixed = [{}]
        lens_sigma = [
            {
                "theta_E": 0.1,
                "gamma": 0.1,
                "e1": 0.1,
                "e2": 0.1,
                "center_x": 0.1,
                "center_y": 0.1,
            }
        ]
        lens_lower = [
            {
                "theta_E": 0.0,
                "gamma": 1.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        lens_upper = [
            {
                "theta_E": 10.0,
                "gamma": 2.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        source_fixed = [{}]
        source_sigma = [
            {
                "R_sersic": 0.05,
                "n_sersic": 0.5,
                "center_x": 0.1,
                "center_y": 0.1,
                "e1": 0.1,
                "e2": 0.1,
            }
        ]
        source_lower = [
            {
                "R_sersic": 0.01,
                "n_sersic": 0.5,
                "center_x": -2,
                "center_y": -2,
                "e1": -0.4,
                "e2": -0.4,
            }
        ]
        source_upper = [
            {
                "R_sersic": 10,
                "n_sersic": 5.5,
                "center_x": 2,
                "center_y": 2,
                "e1": 0.4,
                "e2": 0.4,
            }
        ]

        lens_param = [kwargs_lens, lens_sigma, lens_fixed, lens_lower, lens_upper]
        source_param = [
            kwargs_source,
            source_sigma,
            source_fixed,
            source_lower,
            source_upper,
        ]

        kwargs_params = {"lens_model": lens_param, "source_model": source_param}

        kwargs_constraints = {}

        multi_band_list = [[kwargs_data, kwargs_psf_gaussian, kwargs_numerics]]

        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "multi-linear",
        }

        kwargs_likelihood = {"source_marg": False}

        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []

        kwargs_zeus = {
            "n_burn": 2,
            "n_run": 2,
            "walkerRatio": 4,
            "backend_filename": "test_mcmc_zeus.h5",
        }

        fitting_list.append(["zeus", kwargs_zeus])

        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_multinest(self):
        # Nested sampler tests
        # further decrease the parameter space for nested samplers to run faster

        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        fitting_list = []
        kwargs_update = {
            "ps_add_fixed": [[0, ["ra_source", "dec_source"], [0, 0]]],
            "lens_light_add_fixed": [
                [0, ["n_sersic", "R_sersic", "center_x", "center_y"], [4, 0.1, 0, 0]]
            ],
            "source_add_fixed": [
                [
                    0,
                    ["R_sersic", "e1", "e2", "center_x", "center_y"],
                    [0.6, 0.1, 0.1, 0, 0],
                ]
            ],
            "lens_add_fixed": [
                [
                    0,
                    ["gamma", "theta_E", "e1", "e2", "center_x", "center_y"],
                    [1.8, 1.0, 0.1, 0.1, 0, 0],
                ],
                [1, ["gamma1", "gamma2"], [0.01, 0.01]],
            ],
            "change_source_lower_limit": [[0, ["n_sersic"], [2.9]]],
            "change_source_upper_limit": [[0, ["n_sersic"], [3.1]]],
        }
        fitting_list.append(["update_settings", kwargs_update])
        kwargs_multinest = {
            "kwargs_run": {
                "n_live_points": 10,
                "evidence_tolerance": 0.5,
                "sampling_efficiency": 0.8,  # 1 for posterior-only, 0 for evidence-only
                "importance_nested_sampling": False,
                "multimodal": True,
                "const_efficiency_mode": False,  # reduce sampling_efficiency to 5% when True
            },
            "remove_output_dir": True,
        }

        fitting_list.append(["MultiNest", kwargs_multinest])

        chain_list2 = fittingSequence.fit_sequence(fitting_list)
        kwargs_fixed = fittingSequence._updateManager.fixed_kwargs
        npt.assert_almost_equal(kwargs_fixed[0][1]["gamma1"], 0.01, decimal=2)
        assert fittingSequence._updateManager._lower_kwargs[1][0]["n_sersic"] == 2.9
        assert fittingSequence._updateManager._upper_kwargs[1][0]["n_sersic"] == 3.1

        kwargs_test = {"kwargs_lens": 1}
        fittingSequence.update_state(kwargs_test)
        kwargs_out = fittingSequence.best_fit(bijective=True)
        assert kwargs_out["kwargs_lens"] == 1

    def test_dynesty(self):
        np.random.seed(42)
        kwargs_params = copy.deepcopy(self.kwargs_params)
        kwargs_params["lens_model"][0][0]["theta_E"] += 0.01
        kwargs_params["lens_model"][0][0]["gamma"] += 0.01
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []
        kwargs_dynesty = {
            "kwargs_run": {
                "dlogz_init": 0.01,
                "nlive_init": 20,
                "nlive_batch": 20,
                "maxbatch": 1,
            },
        }

        fitting_list.append(["dynesty", kwargs_dynesty])

        chain_list = fittingSequence.fit_sequence(fitting_list)
        output = chain_list[0]
        assert len(output) == 7
        assert output[0] == "dynesty"
        assert len(output[2]) == output[1].shape[1]
        assert len(output[3]) == len(output[1])

    def test_nautilus(self):
        np.random.seed(42)
        kwargs_params = copy.deepcopy(self.kwargs_params)
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            kwargs_params,
        )

        fitting_list = []
        kwargs_nautilus = {
            "prior_type": "uniform",
            "verbose": True,
            "f_live": 1.0,
            "n_eff": 0.0,
            "n_live": 2,
            "seed": 42,
        }

        fitting_list.append(["Nautilus", kwargs_nautilus])
        chain_list = fittingSequence.fit_sequence(fitting_list)
        output = chain_list[0]
        assert len(output) == 7
        assert output[0] == "Nautilus"
        assert len(output[2]) == output[1].shape[1]
        assert len(output[3]) == len(output[1])
        assert np.isfinite(output[5])
        legacy_output = output[6]
        assert "points" in legacy_output
        assert "log_l" in legacy_output
        assert "log_w" in legacy_output
        assert len(legacy_output["log_l"]) == len(legacy_output["points"])

    def test_dypolychord(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        fitting_list = []
        kwargs_dypolychord = {
            "kwargs_run": {
                "ninit": 8,
                "nlive_const": 10,
                #'seed_increment': 1,
                "resume_dyn_run": False,
                #'init_step': 10,
            },
            "polychord_settings": {
                "seed": 1,
                #'num_repeats': 20
            },
            "dypolychord_dynamic_goal": 0.8,  # 1 for posterior-only, 0 for evidence-only
            "remove_output_dir": True,
        }

        fitting_list.append(["dyPolyChord", kwargs_dypolychord])

        chain_list = fittingSequence.fit_sequence(fitting_list)

    def test_minimizer(self):
        n_p = 2
        n_i = 2

        fitting_list = []

        kwargs_simplex = {"n_iterations": n_i, "method": "Nelder-Mead"}
        fitting_list.append(["SIMPLEX", kwargs_simplex])
        kwargs_simplex = {"n_iterations": n_i, "method": "Powell"}
        fitting_list.append(["SIMPLEX", kwargs_simplex])
        kwargs_pso = {"sigma_scale": 1, "n_particles": n_p, "n_iterations": n_i}
        fitting_list.append(["PSO", kwargs_pso])
        kwargs_mcmc = {"sigma_scale": 1, "n_burn": 1, "n_run": 1, "n_walkers": 10}
        fitting_list.append(["emcee", kwargs_mcmc])
        kwargs_mcmc["re_use_samples"] = True
        kwargs_mcmc["flatten_chains"] = True
        kwargs_mcmc["init_samples"] = np.array(
            [[np.random.normal(1, 0.001)] for i in range(100)]
        )
        fitting_list.append(["emcee", kwargs_mcmc])

        def custom_likelihood(kwargs_lens, **kwargs):
            theta_E = kwargs_lens[0]["theta_E"]
            return -((theta_E - 1.0) ** 2) / 0.1**2 / 2

        kwargs_likelihood = {"custom_logL_addition": custom_likelihood}

        kwargs_data_joint = {"multi_band_list": []}
        kwargs_model = {"lens_model_list": ["SIS"]}
        kwargs_constraints = {}
        lens_param = (
            [{"theta_E": 1, "center_x": 0, "center_y": 0}],
            [{"theta_E": 0.1, "center_x": 0.1, "center_y": 0.1}],
            [{"center_x": 0, "center_y": 0}],
            [{"theta_E": 0, "center_x": -10, "center_y": -10}],
            [{"theta_E": 10, "center_x": 10, "center_y": 10}],
        )

        kwargs_params = {"lens_model": lens_param}
        fittingSequence = FittingSequence(
            kwargs_data_joint,
            kwargs_model,
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_params,
        )
        args = fittingSequence.param_class.kwargs2args(
            kwargs_lens=[{"theta_E": 1, "center_x": 0, "center_y": 0}]
        )
        kwargs_result = fittingSequence.param_class.args2kwargs(args)
        print(kwargs_result)
        print(args, "test args")
        chain_list = fittingSequence.fit_sequence(fitting_list)
        kwargs_result = fittingSequence.best_fit(bijective=False)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"], 1, decimal=2
        )

    def test_best_fit_from_samples(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        param_class = fittingSequence.param_class
        args_ref = np.array(
            param_class.kwargs2args(
                kwargs_lens=self.kwargs_lens,
                kwargs_source=self.kwargs_source,
                kwargs_lens_light=self.kwargs_lens_light,
                kwargs_ps=self.kwargs_ps,
            )
        )
        n_dim = len(args_ref)
        rng = np.random.RandomState(42)

        # Case 1: Flattened
        # samples (n_samples, n_dim), logl (n_samples,)
        samples = args_ref + 0.001 * rng.randn(20, n_dim)
        logl = -rng.uniform(1, 10, size=20)
        samples[7] = args_ref  # set maximum index to args_ref value
        logl[7] = 0.0  # unambiguous maximum
        kwargs_result = fittingSequence.best_fit_from_samples(samples, logl)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"],
            self.kwargs_lens[0]["theta_E"],
            decimal=6,
        )

        # Case 2: Unflattened
        # samples (n_steps, n_walkers, n_dim), logl (n_steps, n_walkers)
        samples = args_ref + 0.001 * rng.randn(5, 6, n_dim)
        logl = -rng.uniform(1, 10, size=(5, 6))
        samples[3, 4] = args_ref  # set maximum index to args_ref value
        logl[3, 4] = 0.0  # unambiguous maximum
        kwargs_result = fittingSequence.best_fit_from_samples(samples, logl)
        npt.assert_almost_equal(
            kwargs_result["kwargs_lens"][0]["theta_E"],
            self.kwargs_lens[0]["theta_E"],
            decimal=6,
        )


if __name__ == "__main__":
    pytest.main()

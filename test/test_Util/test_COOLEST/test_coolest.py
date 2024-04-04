import pytest
import numpy.testing as npt
import numpy as np
import unittest
import os

from lenstronomy.Util.coolest_interface import (
    create_lenstronomy_from_coolest,
    update_coolest_from_lenstronomy,
    create_kwargs_mcmc_from_chain_list,
)
from lenstronomy.Util.coolest_read_util import (
    degree_coolest_to_radian_lenstronomy,
    ellibounds_coolest_to_lenstronomy,
    shearbounds_coolest_to_lenstronomy,
)
from lenstronomy.Util.coolest_update_util import (
    shapelet_amp_lenstronomy_to_coolest,
    folding_coolest,
)

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.image_util as image_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Plots.model_plot import ModelPlot

from lenstronomy.Workflow.fitting_sequence import FittingSequence


TEMPLATE_NAME = "coolest_template"  # name of the base COOLEST template
# INVALID_TEMPLATE_NAME = "invalid_coolest_template"  # name of the COOLEST template that contains errors


class TestCOOLESTinterface(object):
    def test_load(self):
        path = os.getcwd()
        if path[-11:] == "lenstronomy":
            path = os.path.join(path, "test", "test_Util", "test_COOLEST")
        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME), check_external_files=False
        )
        return

    def test_update(self):
        path = os.getcwd()
        if path[-11:] == "lenstronomy":
            path = os.path.join(path, "test", "test_Util", "test_COOLEST")
        kwargs_result = {
            "kwargs_lens": [
                {"gamma1": 0.1, "gamma2": -0.05, "ra_0": 0.0, "dec_0": 0.0},
                {"kappa": 0.2, "ra_0": 0.0, "dec_0": 0.0},
                {
                    "theta_E": 0.7,
                    "e1": -0.15,
                    "e2": 0.01,
                    "center_x": 0.03,
                    "center_y": 0.01,
                },
                {
                    "gamma": 2.03,
                    "theta_E": 0.7,
                    "e1": -0.15,
                    "e2": 0.01,
                    "center_x": 0.03,
                    "center_y": 0.01,
                },
            ],
            "kwargs_source": [
                {
                    "amp": 15.0,
                    "R_sersic": 0.11,
                    "n_sersic": 3.6,
                    "center_x": 0.02,
                    "center_y": -0.03,
                    "e1": 0.1,
                    "e2": -0.2,
                },
                {
                    "amp": np.array(
                        [70.0, 33.0, 2.1, 3.9, 15.0, -16.0, 2.8, -1.7, -4.1, 0.2]
                    ),
                    "n_max": 3,
                    "beta": 0.1,
                    "center_x": 0.1,
                    "center_y": 0.0,
                },
            ],
            "kwargs_lens_light": [
                {
                    "amp": 12.0,
                    "R_sersic": 0.02,
                    "n_sersic": 6.0,
                    "center_x": 0.03,
                    "center_y": 0.01,
                    "e1": 0.0,
                    "e2": -0.15,
                },
            ],
            "kwargs_ps": [
                {
                    "point_amp": np.array([0.1]),
                    "ra_image": np.array([0.25]),
                    "dec_image": np.array([0.2]),
                }
            ],
        }
        update_coolest_from_lenstronomy(
            os.path.join(path, TEMPLATE_NAME),
            kwargs_result,
            ending="_update",
            check_external_files=False,
        )
        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME + "_update"),
            check_external_files=False,
        )
        npt.assert_almost_equal(
            kwargs_out["kwargs_params"]["lens_model"][0][2]["e1"],
            kwargs_result["kwargs_lens"][2]["e1"],
            decimal=4,
        )
        npt.assert_almost_equal(
            kwargs_out["kwargs_params"]["lens_model"][0][2]["e2"],
            kwargs_result["kwargs_lens"][2]["e2"],
            decimal=4,
        )
        # os.remove(os.path.join(path, TEMPLATE_NAME + "_update.json"))

        return

    def test_full(self):
        # use read json ; create an image ; create noise ; do fit (PSO for result + MCMC for chain)
        # create the kwargs mcmc ; upadte json
        path = os.getcwd()
        if path[-11:] == "lenstronomy":
            path = os.path.join(path, "test", "test_Util", "test_COOLEST")

        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME + "_update"),
            check_external_files=False,
        )

        # IMAGE specifics
        background_rms = 0.005  # background noise per pixel
        exp_time = 500.0  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        # PSF : easier for test to create a gaussian PSF
        fwhm = 0.05  # full width at half maximum of PSF
        psf_type = "GAUSSIAN"  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # lensing quantities to create an image
        lens_model_list = kwargs_out["kwargs_model"]["lens_model_list"]
        # parameters of the deflector lens model
        kwargs_sie = {
            "theta_E": 0.66,
            "center_x": 0.05,
            "center_y": 0,
            "e1": -0.1,
            "e2": 0.1,
        }
        kwargs_pemd = {
            "gamma": 2.02,
            "theta_E": 0.66,
            "center_x": 0.05,
            "center_y": 0,
            "e1": -0.1,
            "e2": 0.1,
        }
        # external shear
        kwargs_shear = {
            "gamma1": 0.0,
            "gamma2": -0.05,
        }
        # convergence sheet
        kwargs_conv = {
            "kappa": 0.2,
        }
        kwargs_lens = [kwargs_shear, kwargs_conv, kwargs_sie, kwargs_pemd]
        lens_model_class = LensModel(lens_model_list)

        # Sersic parameters in the initial simulation for the source
        source_model_list = kwargs_out["kwargs_model"]["source_light_model_list"]
        kwargs_sersic_source = {
            "amp": 16,
            "R_sersic": 0.1,
            "n_sersic": 3.5,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.1,
            "center_y": 0,
        }
        kwargs_shapelets_source = {
            "amp": np.array([70.0, 33.0, 2.1, 3.9, 15.0, -16.0, 2.8, -1.7, -4.1, 0.2]),
            "n_max": 3,
            "beta": 0.1,
            "center_x": 0.1,
            "center_y": 0.0,
        }
        kwargs_source = [kwargs_sersic_source, kwargs_shapelets_source]
        source_model_class = LightModel(source_model_list)

        # Sersic parameters in the initial simulation for the lens light
        lens_light_model_list = kwargs_out["kwargs_model"]["lens_light_model_list"]
        kwargs_sersic_lens = {
            "amp": 16,
            "R_sersic": 0.6,
            "n_sersic": 2.5,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.05,
            "center_y": 0,
        }
        kwargs_lens_light = [kwargs_sersic_lens]
        lens_light_model_class = LightModel(lens_light_model_list)

        numPix = 100
        kwargs_out["kwargs_data"]["background_rms"] = background_rms
        kwargs_out["kwargs_data"]["exposure_time"] = exp_time
        kwargs_out["kwargs_data"]["image_data"] = np.zeros((numPix, numPix))
        kwargs_out["kwargs_data"].pop("noise_map")

        data_class = ImageData(**kwargs_out["kwargs_data"])
        # PSF
        pixel_scale = (
            kwargs_out["kwargs_data"]["transform_pix2angle"][1][1]
            / kwargs_out["kwargs_psf"]["point_source_supersampling_factor"]
        )
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": pixel_scale,
            "truncation": 3,
        }
        kwargs_out["kwargs_psf"] = kwargs_psf
        psf_class = PSF(**kwargs_out["kwargs_psf"])

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
        )

        # generate image
        image_model = imageModel.image(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=None,
        )

        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        image_real = image_model + poisson + bkg

        data_class.update_data(image_real)
        kwargs_out["kwargs_data"]["image_data"] = image_real

        # MODELING
        # Notes :
        # All the lines above were meant to create a mock image
        # The following is basically the only lines of code you will need
        # (after running the "create_lenstronomy_from_coolest" function) when you actually do the
        # modeling on a pre-existing image (with associated noise and psf proveded)
        band_list = [
            kwargs_out["kwargs_data"],
            kwargs_out["kwargs_psf"],
            kwargs_numerics,
        ]
        multi_band_list = [band_list]
        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
        }
        kwargs_constraints = {}
        kwargs_likelihood = {"check_bounds": True, "check_positive_flux": True}

        fitting_seq = FittingSequence(
            kwargs_data_joint,
            kwargs_out["kwargs_model"],
            kwargs_constraints,
            kwargs_likelihood,
            kwargs_out["kwargs_params"],
        )

        n_particules = 5
        n_iterations = 5
        wr = 2
        n_run_mcmc = 6
        n_burn_mcmc = 6
        fitting_kwargs_list = [
            [
                "PSO",
                {
                    "sigma_scale": 1.0,
                    "n_particles": n_particules,
                    "n_iterations": n_iterations,
                },
            ],
            [
                "MCMC",
                {
                    "n_burn": n_burn_mcmc,
                    "n_run": n_run_mcmc,
                    "walkerRatio": wr,
                    "sigma_scale": 0.01,
                },
            ],
        ]
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        modelPlot = ModelPlot(
            kwargs_data_joint["multi_band_list"],
            kwargs_out["kwargs_model"],
            kwargs_result,
        )
        kwargs_result.pop("kwargs_tracer_source", None)
        modelPlot._imageModel.image_linear_solve(inv_bool=True, **kwargs_result)
        # the last 2 lines are meant for solving the linear parameters

        # use the function to save mcmc chains in userfriendly mode
        # kwargs_mcmc = create_kwargs_mcmc_from_chain_list(chain_list,kwargs_out['kwargs_model'],kwargs_out['kwargs_params'],
        #                                    kwargs_out['kwargs_data'],kwargs_out['kwargs_psf'],kwargs_numerics,
        #                                    kwargs_constraints,idx_chain=1)

        kwargs_mcmc = create_kwargs_mcmc_from_chain_list(
            chain_list,
            kwargs_out["kwargs_model"],
            kwargs_out["kwargs_params"],
            kwargs_out["kwargs_data"],
            kwargs_out["kwargs_psf"],
            kwargs_numerics,
            kwargs_constraints,
            idx_chain=1,
            likelihood_threshold=-100000,
        )
        # save the results (aka update the COOLEST json)
        update_coolest_from_lenstronomy(
            os.path.join(path, TEMPLATE_NAME),
            kwargs_result,
            kwargs_mcmc,
            check_external_files=False,
        )

        return

    def test_pemd(self):
        path = os.getcwd()
        if path[-11:] == "lenstronomy":
            path = os.path.join(path, "test", "test_Util", "test_COOLEST")
        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME),
            check_external_files=False,
        )
        print(kwargs_out)

        # kwargs_results to update the COOLEST template
        kwargs_result = {
            "kwargs_lens": [
                {
                    "gamma1": 0.0,
                    "gamma2": -0.05,
                },
                {
                    "kappa": 0.2,
                },
                {
                    "theta_E": 0.7,
                    "e1": -0.15,
                    "e2": 0.01,
                    "gamma": 2.1,
                    "center_x": 0.03,
                    "center_y": 0.01,
                },
                {
                    "gamma": 2.02,
                    "theta_E": 0.7,
                    "e1": -0.15,
                    "e2": 0.01,
                    "gamma": 2.1,
                    "center_x": 0.03,
                    "center_y": 0.01,
                },
            ],
            "kwargs_source": [
                {
                    "amp": 15.0,
                    "R_sersic": 0.11,
                    "n_sersic": 3.6,
                    "center_x": 0.02,
                    "center_y": -0.03,
                    "e1": 0.1,
                    "e2": -0.2,
                },
                {
                    "amp": np.array(
                        [70.0, 33.0, 2.1, 3.9, 15.0, -16.0, 2.8, -1.7, -4.1, 0.2]
                    ),
                    "n_max": 3,
                    "beta": 0.1,
                    "center_x": 0.1,
                    "center_y": 0.0,
                },
            ],
            "kwargs_lens_light": [
                {
                    "amp": 11.0,
                    "R_sersic": 0.2,
                    "n_sersic": 3.0,
                    "center_x": 0.03,
                    "center_y": 0.01,
                    "e1": -0.15,
                    "e2": 0.01,
                },
            ],
        }
        # kwargs_mcmc to update the COOLEST template. In real cases, this list would be much bigger
        # as each element is a result from a given point at a given iteration of a MCMC chain

        kwargs_mcmc = {
            "args_lens": [
                [
                    {
                        "gamma1": 0.0,
                        "gamma2": -0.05,
                    },
                    {
                        "kappa": 0.2,
                    },
                    {
                        "theta_E": 0.7,
                        "e1": -0.15,
                        "e2": 0.01,
                        "gamma": 2.1,
                        "center_x": 0.03,
                        "center_y": 0.01,
                    },
                    {
                        "gamma": 2.02,
                        "theta_E": 0.7,
                        "e1": -0.15,
                        "e2": 0.01,
                        "gamma": 2.1,
                        "center_x": 0.03,
                        "center_y": 0.01,
                    },
                ],
                [
                    {
                        "gamma1": 0.0,
                        "gamma2": -0.05,
                    },
                    {
                        "kappa": 0.2,
                    },
                    {
                        "theta_E": 0.7,
                        "e1": -0.15,
                        "e2": 0.01,
                        "gamma": 2.1,
                        "center_x": 0.03,
                        "center_y": 0.01,
                    },
                    {
                        "gamma": 2.02,
                        "theta_E": 0.7,
                        "e1": -0.15,
                        "e2": 0.01,
                        "gamma": 2.1,
                        "center_x": 0.03,
                        "center_y": 0.01,
                    },
                ],
            ],
            "args_source": [
                [
                    {
                        "amp": 15.0,
                        "R_sersic": 0.11,
                        "n_sersic": 3.6,
                        "center_x": 0.02,
                        "center_y": -0.03,
                        "e1": 0.1,
                        "e2": -0.2,
                    },
                    {
                        "amp": np.array(
                            [70.0, 33.0, 2.1, 3.9, 15.0, -16.0, 2.8, -1.7, -4.1, 0.2]
                        ),
                        "n_max": 3,
                        "beta": 0.1,
                        "center_x": 0.1,
                        "center_y": 0.0,
                    },
                ],
                [
                    {
                        "amp": 15.0,
                        "R_sersic": 0.11,
                        "n_sersic": 3.6,
                        "center_x": 0.02,
                        "center_y": -0.03,
                        "e1": 0.1,
                        "e2": -0.2,
                    },
                    {
                        "amp": np.array(
                            [70.0, 33.0, 2.1, 3.9, 15.0, -16.0, 2.8, -1.7, -4.1, 0.2]
                        ),
                        "n_max": 3,
                        "beta": 0.1,
                        "center_x": 0.1,
                        "center_y": 0.0,
                    },
                ],
            ],
            "args_lens_light": [
                [
                    {
                        "amp": 11.0,
                        "R_sersic": 0.2,
                        "n_sersic": 3.0,
                        "center_x": 0.03,
                        "center_y": 0.01,
                        "e1": -0.15,
                        "e2": 0.01,
                    },
                ],
                [
                    {
                        "amp": 11.0,
                        "R_sersic": 0.2,
                        "n_sersic": 3.0,
                        "center_x": 0.03,
                        "center_y": 0.01,
                        "e1": -0.15,
                        "e2": 0.01,
                    },
                ],
            ],
        }
        update_coolest_from_lenstronomy(
            os.path.join(path, TEMPLATE_NAME),
            kwargs_result,
            ending="_update",
            kwargs_mcmc=kwargs_mcmc,
            check_external_files=False,
        )
        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME + "_update"),
            check_external_files=False,
        )
        print(kwargs_out)
        npt.assert_almost_equal(
            kwargs_out["kwargs_params"]["lens_model"][0][2]["e1"],
            kwargs_result["kwargs_lens"][2]["e1"],
            decimal=4,
        )
        npt.assert_almost_equal(
            kwargs_out["kwargs_params"]["lens_model"][0][2]["e2"],
            kwargs_result["kwargs_lens"][2]["e2"],
            decimal=4,
        )

        return

    def test_pemd_via_epl(self):
        path = os.getcwd()
        if path[-11:] == "lenstronomy":
            path = os.path.join(path, "test", "test_Util", "test_COOLEST")
        kwargs_out = create_lenstronomy_from_coolest(
            os.path.join(path, TEMPLATE_NAME),
            use_epl=True,
            check_external_files=False,
        )
        print(kwargs_out)
        assert kwargs_out["kwargs_model"]["lens_model_list"][3] == "EPL"
        # the rest of the test would be identical to test_pemd()

    def test_util_functions(self):
        radian = degree_coolest_to_radian_lenstronomy(None)
        radian = degree_coolest_to_radian_lenstronomy(-120.0)
        npt.assert_almost_equal(radian, np.pi / 6.0, decimal=4)

        radian = degree_coolest_to_radian_lenstronomy(120.0)
        npt.assert_almost_equal(radian, 5 * np.pi / 6.0, decimal=4)

        ellibounds_coolest_to_lenstronomy(0.6, 1.0, None, None)
        shearbounds_coolest_to_lenstronomy(0.0, 0.1, -90.0, None)

        shapelet_amp_lenstronomy_to_coolest(None)
        folding_coolest(np.array([-95.0, 95.0]))
        folding_coolest(-95.0)
        folding_coolest(95.0)

        return

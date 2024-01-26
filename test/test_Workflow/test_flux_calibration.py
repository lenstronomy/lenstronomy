__author__ = "sibirrer"

import copy

import numpy.testing as npt
import numpy as np
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestImageCalibration(object):
    def setup_method(self):
        np.random.seed(41)
        # data specifics
        sigma_bkg = 0.01  # background noise per pixel
        exp_time = 1000  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 20  # cutout pixel size
        deltaPix = 0.2  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.1  # full width half max of PSF

        # PSF specification

        self.kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**self.kwargs_data)
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
            "truncation": 3,
        }
        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {"supersampling_factor": 1}

        self.flux_scale_factor = 0.1
        self.kwargs_lens_light = [
            {"amp": 100.0, "R_sersic": 0.5, "n_sersic": 2, "center_x": 0, "center_y": 0}
        ]
        self.kwargs_lens_light2 = [
            {
                "amp": 100 * self.flux_scale_factor,
                "R_sersic": 0.5,
                "n_sersic": 2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC"]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

        imageModel = ImageModel(
            data_class, psf_class, lens_light_model_class=lens_light_model_class
        )
        image_sim = sim_util.simulate_simple(
            imageModel, kwargs_lens_light=self.kwargs_lens_light
        )
        image_sim2 = sim_util.simulate_simple(
            imageModel, kwargs_lens_light=self.kwargs_lens_light2
        )
        self.kwargs_data["image_data"] = image_sim

        self.kwargs_data2 = copy.deepcopy(self.kwargs_data)
        self.kwargs_data2["image_data"] = image_sim2

        image_band1 = [self.kwargs_data, kwargs_psf, kwargs_numerics]
        image_band2 = [self.kwargs_data2, kwargs_psf, kwargs_numerics]
        multi_band_list = [image_band1, image_band2]
        self.kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "joint-linear",
        }

        self.kwargs_model = {"lens_light_model_list": lens_light_model_list}
        self.kwargs_constraints = {}
        self.kwargs_likelihood = {}
        lens_light_sigma = [
            {"R_sersic": 0.05, "n_sersic": 0.5, "center_x": 0.1, "center_y": 0.1}
        ]
        lens_light_lower = [
            {"R_sersic": 0.01, "n_sersic": 0.5, "center_x": -2, "center_y": -2}
        ]
        lens_light_upper = [
            {"R_sersic": 10, "n_sersic": 5.5, "center_x": 2, "center_y": 2}
        ]
        lens_light_param = (
            self.kwargs_lens_light,
            lens_light_sigma,
            [{}],
            lens_light_lower,
            lens_light_upper,
        )
        self.kwargs_params = {"lens_light_model": lens_light_param}

    def test_flux_calibration(self):
        fittingSequence = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )
        fitting_list = []

        kwargs_calibrate = {
            "n_particles": 20,
            "n_iterations": 40,
            "calibrate_bands": [False, True],
        }
        fitting_list.append(["calibrate_images", kwargs_calibrate])
        chain_list = fittingSequence.fit_sequence(fitting_list)
        multi_band_list_new = fittingSequence.multi_band_list
        kwargs_data2_new = multi_band_list_new[1][0]
        flux_scaling = kwargs_data2_new["flux_scaling"]
        npt.assert_almost_equal(flux_scaling, self.flux_scale_factor, decimal=1)

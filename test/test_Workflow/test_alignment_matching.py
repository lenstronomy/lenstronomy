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


class TestAlignmentMatching(object):
    def setup(self):
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
        transform_pix2angle = self.kwargs_data["transform_pix2angle"]
        self.kwargs_data2 = copy.deepcopy(self.kwargs_data)
        self.delta_x_offset = 0.2
        self.kwargs_data2["ra_at_xy_0"] += self.delta_x_offset
        self.phi_rot = 0.1
        cos_phi, sin_phi = np.cos(self.phi_rot), np.sin(self.phi_rot)
        rot_matrix = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
        transform_pix2angle_rot = np.dot(transform_pix2angle, rot_matrix)
        self.kwargs_data2["transform_pix2angle"] = transform_pix2angle_rot

        data_class = ImageData(**self.kwargs_data)
        data_class2 = ImageData(**self.kwargs_data2)
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
            "truncation": 3,
        }
        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {"supersampling_factor": 1}

        self.kwargs_lens_light = [
            {
                "amp": 100.0,
                "R_sersic": 0.5,
                "n_sersic": 2,
                "e1": 0.3,
                "e2": -0.2,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        lens_light_model_list = ["SERSIC_ELLIPSE"]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

        imageModel = ImageModel(
            data_class, psf_class, lens_light_model_class=lens_light_model_class
        )
        imageModel2 = ImageModel(
            data_class2, psf_class, lens_light_model_class=lens_light_model_class
        )
        image_sim = sim_util.simulate_simple(
            imageModel, kwargs_lens_light=self.kwargs_lens_light
        )
        image_sim2 = sim_util.simulate_simple(
            imageModel2, kwargs_lens_light=self.kwargs_lens_light
        )
        self.kwargs_data["image_data"] = image_sim

        self.kwargs_data2_offset = copy.deepcopy(self.kwargs_data)
        self.kwargs_data2_offset["image_data"] = image_sim2

        image_band1 = [self.kwargs_data, kwargs_psf, kwargs_numerics]
        image_band2 = [self.kwargs_data2_offset, kwargs_psf, kwargs_numerics]
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

        kwargs_align = {
            "n_particles": 20,
            "n_iterations": 40,
            "compute_bands": [False, True],
            "align_offset": True,
            "align_rotation": True,
            "delta_shift": 0.3,
            "delta_rot": 0.5,
        }

        fitting_list.append(["align_images", kwargs_align])
        chain_list = fittingSequence.fit_sequence(fitting_list)
        multi_band_list_new = fittingSequence.multi_band_list
        kwargs_data2_new = multi_band_list_new[1][0]
        ra_shift = kwargs_data2_new["ra_shift"]
        npt.assert_almost_equal(ra_shift, self.delta_x_offset, decimal=1)
        phi_rot = kwargs_data2_new["phi_rot"]
        npt.assert_almost_equal(phi_rot, self.phi_rot, decimal=1)

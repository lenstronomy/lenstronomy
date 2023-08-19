__author__ = "sibirrer"

import pytest
import numpy as np
import os
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.sampler import Sampler, choose_pool
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestSampler(object):
    """Test the fitting sequences."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": deltaPix,
        }
        psf = PSF(**kwargs_psf_gaussian)
        kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": psf.kernel_point_source,
        }
        psf_class = PSF(**kwargs_psf)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_model_list = ["SPEP"]
        self.kwargs_lens = [kwargs_spemd]
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

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
            "compute_mode": "regular",
        }
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel, self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light
        )

        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim
        kwargs_data_joint = {
            "multi_band_list": [[kwargs_data, kwargs_psf, kwargs_numerics]],
            "multi_band_type": "single-band",
        }
        self.data_class = data_class
        self.psf_class = psf_class

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "fixed_magnification_list": [False],
        }
        self.kwargs_numerics = {"subgrid_res": 1, "psf_subgrid": False}

        kwargs_constraints = {
            "image_plane_source_list": [False] * len(source_model_list)
        }

        kwargs_likelihood = {
            "source_marg": False,
            "image_position_uncertainty": 0.004,
            "check_matched_source_position": False,
            "source_position_tolerance": 0.001,
            "source_position_sigma": 0.001,
        }
        self.param_class = Param(kwargs_model, **kwargs_constraints)
        self.Likelihood = LikelihoodModule(
            kwargs_data_joint=kwargs_data_joint,
            kwargs_model=kwargs_model,
            param_class=self.param_class,
            **kwargs_likelihood
        )
        self.sampler = Sampler(likelihoodModule=self.Likelihood)

    def test_pso(self):
        n_particles = 2
        n_iterations = 2
        result, chain = self.sampler.pso(
            n_particles,
            n_iterations,
            lower_start=None,
            upper_start=None,
            threadCount=1,
            init_pos=None,
            mpi=False,
            print_key="PSO",
        )

        assert len(result) == 16

    def test_mcmc_emcee(self):
        n_walkers = 36
        n_run = 2
        n_burn = 2
        mean_start = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        sigma_start = np.ones_like(mean_start) * 0.1
        samples, dist = self.sampler.mcmc_emcee(
            n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=False
        )
        assert len(samples) == n_walkers * n_run
        assert len(dist) == len(samples)

        # test of backup file
        # 1) run a chain specifiying a backup file name
        backup_filename = "test_mcmc_emcee.h5"
        samples_1, dist_1 = self.sampler.mcmc_emcee(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            mpi=False,
            backend_filename=backup_filename,
        )
        assert len(samples_1) == n_walkers * n_run
        # 2) run a chain starting from the backup of previous run
        samples_2, dist_2 = self.sampler.mcmc_emcee(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            mpi=False,
            backend_filename=backup_filename,
            start_from_backend=True,
        )
        assert len(samples_2) == len(samples_1) + n_walkers * n_run
        assert len(dist_2) == len(samples_2)

        os.remove(backup_filename)  # just remove the backup file created above

    def test_mcmc_zeus(self):
        n_walkers = 36
        n_run = 2
        n_burn = 2
        mean_start = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
        )
        sigma_start = np.ones_like(mean_start) * 0.1
        samples, dist = self.sampler.mcmc_zeus(
            n_walkers, n_run, n_burn, mean_start, sigma_start
        )
        assert len(samples) == n_walkers * n_run
        assert len(dist) == len(samples)

        # test of backup file
        backup_filename = "test_mcmc_zeus.h5"
        samples_1, dist_1 = self.sampler.mcmc_zeus(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            backend_filename=backup_filename,
        )
        assert len(samples_1) == n_walkers * n_run

        os.remove(backup_filename)

        # test callbacks
        autocorrelation_callback = True
        splitr_callback = True
        miniter_callback = True

        samples_ac, dist_ac = self.sampler.mcmc_zeus(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            autocorrelation_callback=autocorrelation_callback,
        )
        assert len(samples_ac) == n_walkers * n_run

        # the default nsplits is 2, here we set it to 1 because the test chain is very short
        # i.e. 4/3 != integer; 4/2 = integer
        samples_sp, dist_sp = self.sampler.mcmc_zeus(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            splitr_callback=splitr_callback,
            nsplits=1,
        )
        assert len(samples_sp) == n_walkers * n_run

        samples_mi, dist_mi = self.sampler.mcmc_zeus(
            n_walkers,
            n_run,
            n_burn,
            mean_start,
            sigma_start,
            miniter_callback=miniter_callback,
        )
        assert len(samples_mi) == n_walkers * n_run


if __name__ == "__main__":
    pytest.main()

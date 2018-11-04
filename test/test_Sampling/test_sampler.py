__author__ = 'sibirrer'

import pytest
import numpy as np
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.sampler import Sampler
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF


class TestFittingSequence(object):
    """
    test the fitting sequences
    """

    def setup(self):
        self.SimAPI = Simulation()

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = Data(kwargs_data)
        kwargs_psf = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=11, deltaPix=deltaPix,
                                              truncate=3,
                                              kernel=None)
        kwargs_psf = self.SimAPI.psf_configure(psf_type='PIXEL', fwhm=fwhm, kernelsize=11, deltaPix=deltaPix,
                                              truncate=6,
                                              kernel=kwargs_psf['kernel_point_source'])
        psf_class = PSF(kwargs_psf)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': 0.1, 'e2': 0.1}

        lens_model_list = ['SPEP']
        self.kwargs_lens = [kwargs_spemd]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 3, 'center_x': 0, 'center_y': 0,
                                 'e1': 0.1, 'e2': 0.1}

        lens_light_model_list = ['SERSIC']
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        kwargs_numerics = {'subgrid_res': 1, 'psf_subgrid': False}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, kwargs_numerics=kwargs_numerics)
        image_sim = self.SimAPI.simulate(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light)

        data_class.update_data(image_sim)
        self.data_class = data_class
        self.psf_class = psf_class

        kwargs_model = {'lens_model_list': lens_model_list,
                             'source_light_model_list': source_model_list,
                             'lens_light_model_list': lens_light_model_list,
                             'fixed_magnification_list': [False],
                             }
        self.kwargs_numerics = {
            'subgrid_res': 1,
            'psf_subgrid': False}

        num_source_model = len(source_model_list)

        kwargs_constraints = {'joint_center_lens_light': False,
                                   'joint_center_source_light': False,
                                   'additional_images_list': [False],
                                   'fix_to_point_source_list': [False] * num_source_model,
                                   'image_plane_source_list': [False] * num_source_model,
                                   'solver': False,
                                   }

        kwargs_likelihood = {
                                  'source_marg': True,
                                  'point_source_likelihood': False,
                                  'position_uncertainty': 0.004,
                                  'check_solver': False,
                                  'solver_tolerance': 0.001,
                                  }
        self.param_class = Param(kwargs_model, kwargs_constraints)
        self.Likelihood = LikelihoodModule(imSim_class=imageModel, param_class=self.param_class, kwargs_likelihood=kwargs_likelihood)
        self.sampler = Sampler(likelihoodModule=self.Likelihood)

    def test_pso(self):
        n_particles = 2
        n_iterations = 2
        result, chain = self.sampler.pso(n_particles, n_iterations, lower_start=None, upper_start=None, threadCount=1, init_pos=None,
            mpi=False, print_key='PSO')

        assert len(result) == 16

    def test_mcmc_emcee(self):
        n_walkers = 36
        n_run = 2
        n_burn = 2
        mean_start = self.param_class.setParams(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                   kwargs_lens_light=self.kwargs_lens_light)
        sigma_start = np.ones_like(mean_start) * 0.1
        samples = self.sampler.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=False)

        assert len(samples) == n_walkers * n_run

    def test_mcmc_CH(self):
        walkerRatio = 2
        n_run = 2
        n_burn = 2
        mean_start = self.param_class.setParams(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                                kwargs_lens_light=self.kwargs_lens_light)
        sigma_start = np.ones_like(mean_start) * 0.1
        self.sampler.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi=False)


if __name__ == '__main__':
    pytest.main()

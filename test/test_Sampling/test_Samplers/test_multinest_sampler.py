__author__ = 'aymgal'

import pytest
import os
import shutil
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from lenstronomy.Sampling.Samplers.multinest_sampler import MultiNestSampler

try:
    import pymultinest
except:
    print("Warning : MultiNest/pymultinest not installed properly, \
but tests will be trivially fulfilled")
    pymultinest_installed =  False
else:
    pymultinest_installed =  True


class TestMultiNestSampler(object):
    """
    test the fitting sequences
    """

    def setup(self):

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        kwargs_psf_gaussian = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': deltaPix}
        psf = PSF(**kwargs_psf_gaussian)
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf.kernel_point_source}
        psf_class = PSF(**kwargs_psf)
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

        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False, 'compute_mode': 'regular'}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light)

        data_class.update_data(image_sim)
        kwargs_data['image_data'] = image_sim
        kwargs_data_joint = {'multi_band_list': [[kwargs_data, kwargs_psf, kwargs_numerics]], 'multi_band_type': 'single-band'}
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

        kwargs_constraints = {'image_plane_source_list': [False] * len(source_model_list)}

        kwargs_likelihood = {
                                  'source_marg': True,
                                  'position_uncertainty': 0.004,
                                  'check_solver': False,
                                  'solver_tolerance': 0.001,
                                  }
        self.param_class = Param(kwargs_model, **kwargs_constraints)
        self.Likelihood = LikelihoodModule(kwargs_data_joint=kwargs_data_joint, kwargs_model=kwargs_model,
                                           param_class=self.param_class, **kwargs_likelihood)

        prior_means = self.param_class.kwargs2args(kwargs_lens=self.kwargs_lens, kwargs_source=self.kwargs_source,
                                                   kwargs_lens_light=self.kwargs_lens_light)
        prior_sigmas = np.ones_like(prior_means) * 0.1
        self.output_dir = 'test_nested_out'
        self.sampler = MultiNestSampler(self.Likelihood, prior_type='uniform',
                                        prior_means=prior_means, 
                                        prior_sigmas=prior_sigmas,
                                        output_dir=self.output_dir,
                                        remove_output_dir=True)

    def test_sampler(self):
        kwargs_run = {
            'n_live_points': 10,
            'evidence_tolerance': 0.5,
            'sampling_efficiency': 0.8,  # 1 for posterior-only, 0 for evidence-only
            'importance_nested_sampling': False,
            'multimodal': True,
            'const_efficiency_mode': False,   # reduce sampling_efficiency to 5% when True
        }
        samples, means, logZ, logZ_err, logL, results = self.sampler.run(kwargs_run)
        assert len(means) == 16
        if not pymultinest_installed:
            # trivial test when pymultinest is not installed properly
            assert np.count_nonzero(samples) == 0
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_sampler_init(self):
        test_dir = 'some_dir'
        os.mkdir(test_dir)
        sampler = MultiNestSampler(self.Likelihood, prior_type='uniform',
                                   output_dir=test_dir)
        shutil.rmtree(test_dir, ignore_errors=True)
        try:
            sampler = MultiNestSampler(self.Likelihood, prior_type='gaussian',
                                       prior_means=None, # will raise an Error 
                                       prior_sigmas=None, # will raise an Error
                                       output_dir=None,
                                       remove_output_dir=True)
        except Exception as e:
            assert isinstance(e, ValueError)
        try:
            sampler = MultiNestSampler(self.Likelihood, prior_type='some_type')
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_prior(self):
        n_dims = self.sampler.n_dims
        cube_low = np.zeros(n_dims)
        cube_upp = np.ones(n_dims)

        self.prior_type = 'uniform'
        self.sampler.prior(cube_low, n_dims, n_dims)
        npt.assert_equal(cube_low, self.sampler.lowers)
        self.sampler.prior(cube_upp, n_dims, n_dims)
        npt.assert_equal(cube_upp, self.sampler.uppers)

        cube_mid = 0.5 * np.ones(n_dims)
        self.prior_type = 'gaussian'
        self.sampler.prior(cube_mid, n_dims, n_dims)
        cube_gauss = np.array([50., 50., 0., 0., 0., 0., 50., 4.25, 
                                0., 0., 0., 0., 50., 4.25, 0., 0.])
        npt.assert_equal(cube_mid, cube_gauss)

    def test_log_likelihood(self):
        n_dims = self.sampler.n_dims
        args = np.nan * np.ones(n_dims)
        logL = self.sampler.log_likelihood(args, n_dims, n_dims)
        assert logL == -1e15

if __name__ == '__main__':
    pytest.main()

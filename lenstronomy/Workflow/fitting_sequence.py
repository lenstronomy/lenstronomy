from lenstronomy.Workflow.psf_fitting import PsfFitting
from lenstronomy.Sampling.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.alignment_matching import AlignmentFitting
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Workflow.update_manager import UpdateManager
from lenstronomy.Sampling.sampler import Sampler
from lenstronomy.Sampling.likelihood import LikelihoodModule
import numpy as np


class FittingSequence(object):
    """
    class to define a sequence of fitting applied, inherit the Fitting class
    this is a Workflow manager that allows to update model configurations before executing another step in the modelling
    The user can take this module as an example of how to create their own workflows or build their own around the FittingSequence
    """
    def __init__(self, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False, verbose=True):
        self.multi_band_list = multi_band_list
        self._verbose = verbose
        self._mpi = mpi
        self._updateManager = UpdateManager(kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
        self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self._updateManager.init_kwargs

    def fit_sequence(self, fitting_list, fitting_kwargs_list):
        """

        :param fitting_kwargs_list: list of kwargs specify the fitting routine to be executed
        :param bijective: bool, if True, does not map parameters sampled in the image plane to the source plane.
        :return:
        """
        chain_list = []
        param_list = []
        samples_mcmc, param_mcmc, dist_mcmc = [], [], []
        for i, fitting_type in enumerate(fitting_list):
            if fitting_type == 'restart':
                self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self._updateManager.init_kwargs
            elif fitting_type == 'update_settings':
                self.update_settings(**fitting_kwargs_list[i])
            elif fitting_type == 'psf_iteration':
                self.psf_iteration(**fitting_kwargs_list[i])
            elif fitting_type == 'align_images':
                self.align_images(**fitting_kwargs_list[i])
            elif fitting_type == 'PSO':
                lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param = self.pso(**fitting_kwargs_list[i])
                self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = lens_result, source_result, lens_light_result, ps_result, cosmo_result
                chain_list.append(chain)
                param_list.append(param)
            elif fitting_type == 'MCMC':
                samples_mcmc, param_mcmc, dist_mcmc = self.mcmc(**fitting_kwargs_list[i])
            else:
                raise ValueError("fitting_sequence %s is not supported. Please use: 'PSO', 'MCMC', 'psf_iteration', "
                                 "'restart', 'update_settings' or ""'align_images'" % fitting_type)
        return chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc

    def best_fit(self, bijective=False):
        param_class = self._updateManager.param_class(self._lens_temp)
        if bijective is False:
            lens_temp = param_class.update_lens_scaling(self._cosmo_temp, self._lens_temp, inverse=False)
            source_temp = param_class.image2source_plane(self._source_temp, lens_temp)
        else:
            lens_temp, source_temp = self._lens_temp, self._source_temp
        return lens_temp, source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp

    def mcmc(self, n_burn, n_run, walkerRatio, sigma_scale=1, threadCount=1, init_samples=None):
        """

        :param fitting_kwargs:
        :param lens_input:
        :param source_input:
        :param lens_light_input:
        :param ps_input:
        :return:
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        param_class = self._updateManager.param_class(self._lens_temp)

        imSim_class = class_creator.create_multiband(self.multi_band_list, **kwargs_model)
        likelihoodModule = LikelihoodModule(imSim_class=imSim_class, param_class=param_class, **kwargs_likelihood)
        # run PSO
        mcmc_class = Sampler(likelihoodModule=likelihoodModule)
        mean_start = param_class.kwargs2args(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                           self._cosmo_temp)
        lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_samples is None:
            initpos = ReusePositionGenerator(init_samples)
        else:
            initpos = None
        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, np.array(sigma_start) * sigma_scale,
                                           threadCount=threadCount,
                                           mpi=self._mpi, init_pos=initpos)
        return samples, param_list, dist

    def pso(self, n_particles, n_iterations, sigma_scale=1, print_key='PSO', threadCount=1):
        """

        :param fitting_kwargs:
        :param lens_input:
        :param source_input:
        :param lens_light_input:
        :param ps_input:
        :return:
        """

        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        param_class = self._updateManager.param_class(self._lens_temp)
        init_pos = param_class.kwargs2args(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                           self._cosmo_temp)
        lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma)
        lowerLimit = np.array(init_pos) - np.array(sigma_start) * sigma_scale
        upperLimit = np.array(init_pos) + np.array(sigma_start) * sigma_scale
        num_param, param_list = param_class.num_param()

        # initialize ImSim() class
        imSim_class = class_creator.create_multiband(self.multi_band_list, **kwargs_model)
        likelihoodModule = LikelihoodModule(imSim_class=imSim_class, param_class=param_class, **kwargs_likelihood)
        # run PSO
        sampler = Sampler(likelihoodModule=likelihoodModule)
        result, chain = sampler.pso(n_particles, n_iterations, lowerLimit, upperLimit, init_pos=init_pos,
                                       threadCount=threadCount, mpi=self._mpi, print_key=print_key)
        lens_result, source_result, lens_light_result, ps_result, cosmo_result = param_class.args2kwargs(result,
                                                                                                         bijective=True)
        return lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param_list

    def psf_iteration(self, num_iter=10, no_break=True, stacking_method='median', block_center_neighbour=0, keep_psf_error_map=True,
                 psf_symmetry=1, psf_iter_factor=1, verbose=True, compute_bands=None):
        #lens_temp = copy.deepcopy(lens_input)
        kwargs_model = self._updateManager.kwargs_model
        param_class = self._updateManager.param_class(self._lens_temp)
        lens_updated = param_class.update_lens_scaling(self._cosmo_temp, self._lens_temp)
        source_updated = param_class.image2source_plane(self._source_temp, lens_updated)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for i in range(len(self.multi_band_list)):
            if compute_bands[i] is True:
                kwargs_data = self.multi_band_list[i][0]
                kwargs_psf = self.multi_band_list[i][1]
                kwargs_numerics = self.multi_band_list[i][2]
                image_model = class_creator.create_image_model(kwargs_data=kwargs_data,
                                                               kwargs_psf=kwargs_psf,
                                                               kwargs_numerics=kwargs_numerics,
                                                               **kwargs_model)
                psf_iter = PsfFitting(image_model_class=image_model)
                kwargs_psf = psf_iter.update_iterative(kwargs_psf, lens_updated, source_updated,
                                                       self._lens_light_temp, self._ps_temp, num_iter=num_iter,
                                                       no_break=no_break, stacking_method=stacking_method,
                                                       block_center_neighbour=block_center_neighbour,
                                                       keep_psf_error_map=keep_psf_error_map,
                 psf_symmetry=psf_symmetry, psf_iter_factor=psf_iter_factor, verbose=verbose)
                self.multi_band_list[i][1] = kwargs_psf
        return 0

    def align_images(self, n_particles=10, n_iterations=10, lowerLimit=-0.2, upperLimit=0.2, threadCount=1,
                     compute_bands=None):
        kwargs_model = self._updateManager.kwargs_model
        param_class = self._updateManager.param_class(self._lens_temp)
        lens_updated = param_class.update_lens_scaling(self._cosmo_temp, self._lens_temp)
        source_updated = param_class.image2source_plane(self._source_temp, lens_updated)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for i in range(len(self.multi_band_list)):
            if compute_bands[i] is True:
                kwargs_data = self.multi_band_list[i][0]
                kwargs_psf = self.multi_band_list[i][1]
                kwargs_numerics = self.multi_band_list[i][2]
                alignmentFitting = AlignmentFitting(kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model, lens_updated, source_updated,
                                                        self._lens_light_temp, self._ps_temp)

                kwargs_data, chain = alignmentFitting.pso(n_particles=n_particles, n_iterations=n_iterations,
                                                          lowerLimit=lowerLimit, upperLimit=upperLimit,
                                                          threadCount=threadCount, mpi=self._mpi,
                                                          print_key='Alignment fitting for band %s ...' % i)
                print('Align completed for band %s.' % i)
                print('ra_shift: %s,  dec_shift: %s' %(kwargs_data['ra_shift'], kwargs_data['dec_shift']))
                self.multi_band_list[i][0] = kwargs_data
        return 0

    def update_settings(self, kwargs_model={}, kwargs_constraints={}, kwargs_likelihood={}, lens_add_fixed=[],
                     source_add_fixed=[], lens_light_add_fixed=[], ps_add_fixed=[], cosmo_add_fixed=[], lens_remove_fixed=[],
                     source_remove_fixed=[], lens_light_remove_fixed=[], ps_remove_fixed=[], cosmo_remove_fixed=[]):
        self._updateManager.update_options(kwargs_model, kwargs_constraints, kwargs_likelihood)
        self._updateManager.update_fixed(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                         self._cosmo_temp, lens_add_fixed, source_add_fixed, lens_light_add_fixed,
                                         ps_add_fixed, cosmo_add_fixed, lens_remove_fixed, source_remove_fixed,
                                         lens_light_remove_fixed, ps_remove_fixed, cosmo_remove_fixed)
        return 0


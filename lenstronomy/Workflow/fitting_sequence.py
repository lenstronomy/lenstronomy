from lenstronomy.Workflow.psf_fitting import PsfFitting
from lenstronomy.Sampling.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.alignment_matching import AlignmentFitting
from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel
from lenstronomy.Workflow.multi_band_manager import MultiBandUpdateManager
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.sampler import Sampler
from lenstronomy.Sampling.Samplers.multinest_sampler import MultiNestSampler
from lenstronomy.Sampling.Samplers.polychord_sampler import DyPolyChordSampler
from lenstronomy.Sampling.Samplers.dynesty_sampler import DynestySampler
import numpy as np
import sys


class FittingSequence(object):
    """
    class to define a sequence of fitting applied, inherit the Fitting class
    this is a Workflow manager that allows to update model configurations before executing another step in the modelling
    The user can take this module as an example of how to create their own workflows or build their own around the FittingSequence
    """
    def __init__(self, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False,
                 verbose=True):
        """

        :param kwargs_data_joint:
        :param kwargs_model:
        :param kwargs_constraints:
        :param kwargs_likelihood:
        :param kwargs_params:
        :param mpi:
        :param verbose: bool, if True
        """
        self.kwargs_data_joint = kwargs_data_joint
        self.multi_band_list = kwargs_data_joint.get('multi_band_list', [])
        self._verbose = verbose
        self._mpi = mpi
        self._updateManager = MultiBandUpdateManager(kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params,
                                                     num_bands=len(self.multi_band_list))
        self._mcmc_init_samples = None

    def kwargs_fixed(self):
        """
        returns the updated kwargs_fixed from the update Manager

        :return: list of fixed kwargs, see UpdateManager()
        """
        return self._updateManager.fixed_kwargs

    def fit_sequence(self, fitting_list):
        """

        :param fitting_list: list of [['string', {kwargs}], ..] with 'string being the specific fitting option and kwargs being the arguments passed to this option
        :return: fitting results
        """
        chain_list = []
        #param_list = []
        #samples_mcmc, param_mcmc, dist_mcmc = [], [], []
        for i, fitting in enumerate(fitting_list):
            fitting_type = fitting[0]
            kwargs = fitting[1]

            if fitting_type == 'restart':
                self._updateManager.set_init_state()

            elif fitting_type == 'update_settings':
                self.update_settings(**kwargs)

            elif fitting_type == 'fix_not_computed':
                self.fix_not_computed(**kwargs)

            elif fitting_type == 'psf_iteration':
                self.psf_iteration(**kwargs)

            elif fitting_type == 'align_images':
                self.align_images(**kwargs)

            elif fitting_type == 'PSO':
                lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param = self.pso(**kwargs)
                self._updateManager.update_param_state(lens_result, source_result, lens_light_result, ps_result, cosmo_result)
                chain_list.append([fitting_type, chain, param])

            elif fitting_type == 'MCMC':
                if not 'init_samples' in kwargs:
                    kwargs['init_samples'] = self._mcmc_init_samples
                elif kwargs['init_samples'] is None:
                    kwargs['init_samples'] = self._mcmc_init_samples
                mcmc_output = self.mcmc(**kwargs)
                chain_list.append(mcmc_output)

            elif fitting_type == 'nested_sampling':
                ns_output = self.nested_sampling(**kwargs)
                chain_list.append(ns_output)

            else:
                raise ValueError("fitting_sequence %s is not supported. Please use: 'PSO', 'MCMC', 'psf_iteration', "
                                 "'restart', 'update_settings' or ""'align_images'" % fitting_type)
        return chain_list

    def best_fit(self, bijective=False):
        """

        :param bijective: bool, if True, the mapping of image2source_plane and the mass_scaling parameterisation are inverted. If you do not use those options, there is no effect.
        :return: best fit model of the current state of the FittingSequence class
        """

        lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp = self._updateManager.parameter_state
        if bijective is False:
            param_class = self._updateManager.param_class
            lens_temp = param_class.update_lens_scaling(cosmo_temp, lens_temp, inverse=False)
            source_temp = param_class.image2source_plane(source_temp, lens_temp)
        return lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp

    @property
    def best_fit_likelihood(self):
        """
        returns the log likelihood of the best fit model of the current state of this class

        :return: log likelihood, float
        """
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo = self.best_fit(bijective=False)
        param_class = self.param_class
        likelihoodModule = self.likelihoodModule
        logL, _ = likelihoodModule.logL(param_class.kwargs2args(kwargs_lens, kwargs_source, kwargs_lens_light,
                                                                kwargs_ps, kwargs_cosmo))
        return logL

    @property
    def param_class(self):
        """

        :return: Param() class instance reflecting the current state of Fittingsequence
        """
        return self._updateManager.param_class

    @property
    def likelihoodModule(self):
        """

        :return: Likelihood() class instance reflecting the current state of Fittingsequence
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihoodModule = LikelihoodModule(self.kwargs_data_joint, kwargs_model, self.param_class, **kwargs_likelihood)
        return likelihoodModule

    def mcmc(self, n_burn, n_run, walkerRatio, sigma_scale=1, threadCount=1, init_samples=None, re_use_samples=True,
             sampler_type='COSMOHAMMER'):
        """
        MCMC routine

        :param n_burn: number of burn in iterations (will not be saved)
        :param n_run: number of MCMC iterations that are saved
        :param walkerRatio: ratio of walkers/number of free parameters
        :param sigma_scale: scaling of the initial parameter spread relative to the width in the initial settings
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :param init_samples: initial sample from where to start the MCMC process
        :param re_use_samples: bool, if True, re-uses the samples described in init_samples.nOtherwise starts from scratch.
        :param sampler_type: string, which MCMC sampler to be used. Options are: 'COSMOHAMMER, and 'EMCEE'
        :return: list of output arguments, e.g. MCMC samples, parameter names, logL distances of all samples specified by the specific sampler used
        """

        param_class = self.param_class
        # run PSO
        mcmc_class = Sampler(likelihoodModule=self.likelihoodModule)
        lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp = self._updateManager.parameter_state
        mean_start = param_class.kwargs2args(lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp)
        lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_samples is None and re_use_samples is True:
            print("test that you are here!")
            num_samples, num_param_prev = np.shape(init_samples)
            print(num_samples, num_param_prev, num_param, 'shape of init_sample')
            if num_param_prev == num_param:
                print("re-using previous samples to initialize the next MCMC run.")
                initpos = ReusePositionGenerator(init_samples)
            else:
                print("Can not re-use previous MCMC samples due to change in option")
                initpos = None
        else:
            initpos = None
        if sampler_type is 'COSMOHAMMER':
            samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, np.array(sigma_start) * sigma_scale,
                                           threadCount=threadCount,
                                           mpi=self._mpi, init_pos=initpos)
            output = [sampler_type, samples, param_list, dist]
        elif sampler_type is 'EMCEE':
            n_walkers = num_param * walkerRatio
            samples = mcmc_class.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=self._mpi)
            output = [sampler_type, samples, param_list]
        else:
            raise ValueError('sampler_type %s not supported!' % sampler_type)
        self._mcmc_init_samples = samples  # overwrites previous samples to continue from there in the next MCMC run
        return output

    def pso(self, n_particles, n_iterations, sigma_scale=1, print_key='PSO', threadCount=1):
        """
        Particle Swarm Optimization

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param sigma_scale: scaling of the initial parameter spread relative to the width in the initial settings
        :param print_key: string, printed text when executing this routine
        :param threadCount: number of CPU threads. If MPI option is set, threadCount=1
        :return: result of the best fit, the chain of the best fit parameter after each iteration, list of parameters in same order
        """

        param_class = self.param_class
        lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp = self._updateManager.parameter_state
        init_pos = param_class.kwargs2args(lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp)
        lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma = self._updateManager.sigma_kwargs
        sigma_start = param_class.kwargs2args(lens_sigma, source_sigma, lens_light_sigma, ps_sigma, cosmo_sigma)
        lowerLimit = np.array(init_pos) - np.array(sigma_start) * sigma_scale
        upperLimit = np.array(init_pos) + np.array(sigma_start) * sigma_scale
        num_param, param_list = param_class.num_param()

        # run PSO
        sampler = Sampler(likelihoodModule=self.likelihoodModule)
        result, chain = sampler.pso(n_particles, n_iterations, lowerLimit, upperLimit, init_pos=init_pos,
                                       threadCount=threadCount, mpi=self._mpi, print_key=print_key)
        lens_result, source_result, lens_light_result, ps_result, cosmo_result = param_class.args2kwargs(result,
                                                                                                         bijective=True)
        return lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param_list

    def nested_sampling(self, sampler_type='MULTINEST', kwargs_run={},
                        prior_type='uniform', width_scale=1, sigma_scale=1, 
                        output_basename='chain', remove_output_dir=True, 
                        dypolychord_dynamic_goal=0.8,
                        output_dir="nested_sampling_chains",
                        dynesty_bound='multi', dynesty_sample='auto'):
        """
        Run (Dynamic) Nested Sampling algorithms, depending on the type of algorithm.

        :param sampler_type: 'MULTINEST', 'DYPOLYCHORD', 'DYNESTY'
        :param kwargs_run: keywords passed to the core sampling method
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param width_scale: scale the width (lower/upper limits) of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param output_basename: name of the folder in which the core MultiNest/PolyChord code will save output files
        :param remove_output_dir: if True, the above folder is removed after completion
        :param dypolychord_dynamic_goal: dynamic goal for DyPolyChord (trade-off between evidence (0) and posterior (1) computation)
        :param dynesty_bound: see https://dynesty.readthedocs.io for details
        :param dynesty_sample: see https://dynesty.readthedocs.io for details
        :return: list of output arguments : samples, mean inferred values, log-likelihood, log-evidence, error on log-evidence for each sample
        """
        mean_start, sigma_start = self._prepare_sampling(prior_type)

        if sampler_type == 'MULTINEST':
            sampler = MultiNestSampler(self.likelihoodModule,
                                       prior_type=prior_type,
                                       prior_means=mean_start,
                                       prior_sigmas=sigma_start,
                                       width_scale=width_scale,
                                       sigma_scale=sigma_scale,
                                       output_dir=output_dir,
                                       output_basename=output_basename,
                                       remove_output_dir=remove_output_dir,
                                       use_mpi=self._mpi)
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(kwargs_run)

        elif sampler_type == 'DYPOLYCHORD':
            sampler = DyPolyChordSampler(self.likelihoodModule,
                                         prior_type=prior_type,
                                         prior_means=mean_start,
                                         prior_sigmas=sigma_start,
                                         width_scale=width_scale,
                                         sigma_scale=sigma_scale,
                                         output_dir=output_dir,
                                         output_basename=output_basename,
                                         remove_output_dir=remove_output_dir,
                                         use_mpi=self._mpi)
            samples, means, logZ, logZ_err, logL, results_object \
                = sampler.run(dypolychord_dynamic_goal, kwargs_run)

        elif sampler_type == 'DYNESTY':
            sampler = DynestySampler(self.likelihoodModule,
                                     prior_type=prior_type,
                                     prior_means=mean_start,
                                     prior_sigmas=sigma_start,
                                     width_scale=width_scale,
                                     sigma_scale=sigma_scale,
                                     bound=dynesty_bound, 
                                     sample=dynesty_sample,
                                     use_mpi=self._mpi)
            samples, means, logZ, logZ_err, logL, results_object = sampler.run(kwargs_run)

        else:
            raise ValueError('Sampler type %s not supported.' % sampler_type)
        # update current best fit values
        self._update_state(means)

        output = [sampler_type, samples, sampler.param_names, logL, 
                  logZ, logZ_err, results_object]
        return output

    def psf_iteration(self, num_iter=10, no_break=True, stacking_method='median', block_center_neighbour=0,
                      keep_psf_error_map=True, psf_symmetry=1, psf_iter_factor=1, verbose=True, compute_bands=None):
        """
        iterative PSF reconstruction

        :param num_iter: number of iterations in the process
        :param no_break: bool, if False will break the process as soon as one step lead to a wors reconstruction then the previous step
        :param stacking_method: string, 'median' and 'mean' supported
        :param block_center_neighbour: radius of neighbouring point source to be blocked in the reconstruction
        :param keep_psf_error_map: bool, whether or not to keep the previous psf_error_map
        :param psf_symmetry: int, number of invariant rotations in the reconstructed PSF
        :param psf_iter_factor: factor of new estimated PSF relative to the old one PSF_updated = (1-psf_iter_factor) * PSF_old + psf_iter_factor*PSF_new
        :param verbose: bool, print statements
        :param compute_bands: bool list, if multiple bands, this process can be limited to a subset of bands
        :return: 0, updated PSF is stored in self.mult_iband_list
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get('image_likelihood_mask_list', None)
        param_class = self.param_class
        lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp = self._updateManager.parameter_state
        lens_updated = param_class.update_lens_scaling(cosmo_temp, lens_temp)
        source_updated = param_class.image2source_plane(source_temp, lens_updated)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for band_index in range(len(self.multi_band_list)):
            if compute_bands[band_index] is True:
                kwargs_psf = self.multi_band_list[band_index][1]
                image_model = SingleBandMultiModel(self.multi_band_list, kwargs_model,
                                                   likelihood_mask_list=likelihood_mask_list, band_index=band_index)
                psf_iter = PsfFitting(image_model_class=image_model)
                kwargs_psf = psf_iter.update_iterative(kwargs_psf, lens_updated, source_updated,
                                                       lens_light_temp, ps_temp, num_iter=num_iter,
                                                       no_break=no_break, stacking_method=stacking_method,
                                                       block_center_neighbour=block_center_neighbour,
                                                       keep_psf_error_map=keep_psf_error_map,
                 psf_symmetry=psf_symmetry, psf_iter_factor=psf_iter_factor, verbose=verbose)
                self.multi_band_list[band_index][1] = kwargs_psf
        return 0

    def align_images(self, n_particles=10, n_iterations=10, lowerLimit=-0.2, upperLimit=0.2, threadCount=1,
                     compute_bands=None):
        """
        aligns the coordinate systems of different exposures within a fixed model parameterisation by executing a PSO
        with relative coordinate shifts as free parameters

        :param n_particles: number of particles in the Particle Swarm Optimization
        :param n_iterations: number of iterations in the optimization process
        :param lowerLimit: lower limit of relative shift
        :param upperLimit: upper limit of relative shift
        :param verbose: bool, print statements
        :param compute_bands: bool list, if multiple bands, this process can be limited to a subset of bands
        :return:
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        likelihood_mask_list = kwargs_likelihood.get('image_likelihood_mask_list', None)
        param_class = self.param_class
        lens_temp, source_temp, lens_light_temp, ps_temp, cosmo_temp = self._updateManager.parameter_state
        lens_updated = param_class.update_lens_scaling(cosmo_temp, lens_temp)
        source_updated = param_class.image2source_plane(source_temp, lens_updated)
        if compute_bands is None:
            compute_bands = [True] * len(self.multi_band_list)

        for i in range(len(self.multi_band_list)):
            if compute_bands[i] is True:

                alignmentFitting = AlignmentFitting(self.multi_band_list, kwargs_model, lens_updated, source_updated,
                                                        lens_light_temp, ps_temp, band_index=i,
                                                    likelihood_mask_list=likelihood_mask_list)

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
                     source_remove_fixed=[], lens_light_remove_fixed=[], ps_remove_fixed=[], cosmo_remove_fixed=[],
                        change_source_lower_limit=None, change_source_upper_limit=None):
        """
        updates lenstronomy settings "on the fly"

        :param kwargs_model: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_constraints: kwargs, specified keyword arguments overwrite the existing ones
        :param kwargs_likelihood: kwargs, specified keyword arguments overwrite the existing ones
        :param lens_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_add_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param cosmo_add_fixed: ['param1', 'param2',...]
        :param lens_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param source_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param lens_light_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param ps_remove_fixed: [[i_model, ['param1', 'param2',...], [...]]
        :param cosmo_remove_fixed: ['param1', 'param2',...]
        :return: 0, the settings are overwritten for the next fitting step to come
        """
        self._updateManager.update_options(kwargs_model, kwargs_constraints, kwargs_likelihood)
        self._updateManager.update_fixed(lens_add_fixed, source_add_fixed, lens_light_add_fixed,
                                         ps_add_fixed, cosmo_add_fixed, lens_remove_fixed, source_remove_fixed,
                                         lens_light_remove_fixed, ps_remove_fixed, cosmo_remove_fixed)
        self._updateManager.update_limits(change_source_lower_limit, change_source_upper_limit)
        return 0

    def fix_not_computed(self, free_bands):
        """
        fixes lens model parameters of imaging bands/frames that are not computed and frees the parameters of the other
        lens models to the initial kwargs_fixed options

        :param free_bands: bool list of length of imaging bands in order of imaging bands, if False: set fixed lens model
        :return: None
        """
        self._updateManager.fix_not_computed(free_bands=free_bands)

    def _prepare_sampling(self, prior_type):
        if prior_type == 'gaussian':
            mean_start = self.param_class.kwargs2args(*self._updateManager.parameter_state)
            sigma_start = self.param_class.kwargs2args(*self._updateManager.sigma_kwargs)
            mean_start  = np.array(mean_start)
            sigma_start = np.array(sigma_start)
        else:
            mean_start, sigma_start = None, None
        return mean_start, sigma_start

    def _update_state(self, result):
        """

        :param result: array of parameters being sampled (e.g. result of MCMC chain)
        :return: None, updates the parameter state of the class instance
        """
        lens_result, source_result, lens_light_result, ps_result, cosmo_result \
            = self.param_class.args2kwargs(result, bijective=True)

        self._updateManager.update_param_state(lens_result, source_result, lens_light_result, ps_result, cosmo_result)

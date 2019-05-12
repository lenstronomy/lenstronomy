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
    def __init__(self, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False,
                 verbose=True):
        """

        :param multi_band_list:
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
        self._updateManager = UpdateManager(kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
        self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self._updateManager.init_kwargs
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
        param_list = []
        samples_mcmc, param_mcmc, dist_mcmc = [], [], []
        for i, fitting in enumerate(fitting_list):
            fitting_type = fitting[0]
            kwargs = fitting[1]
            if fitting_type == 'restart':
                self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = self._updateManager.init_kwargs
            elif fitting_type == 'update_settings':
                self.update_settings(**kwargs)
            elif fitting_type == 'psf_iteration':
                self.psf_iteration(**kwargs)
            elif fitting_type == 'align_images':
                self.align_images(**kwargs)
            elif fitting_type == 'PSO':
                lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param = self.pso(**kwargs)
                self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp = lens_result, source_result, lens_light_result, ps_result, cosmo_result
                chain_list.append(chain)
                param_list.append(param)
            elif fitting_type == 'MCMC':
                if not 'init_samples' in kwargs:
                    kwargs['init_samples'] = self._mcmc_init_samples
                elif kwargs['init_samples'] is None:
                    kwargs['init_samples'] = self._mcmc_init_samples
                samples_mcmc, param_mcmc, dist_mcmc = self.mcmc(**kwargs)
                self._mcmc_init_samples = samples_mcmc
            else:
                raise ValueError("fitting_sequence %s is not supported. Please use: 'PSO', 'MCMC', 'psf_iteration', "
                                 "'restart', 'update_settings' or ""'align_images'" % fitting_type)
        return chain_list, param_list, samples_mcmc, param_mcmc, dist_mcmc

    def best_fit(self, bijective=False):
        """

        :param bijective: bool, if True, the mapping of image2source_plane and the mass_scaling parameterisation are inverted. If you do not use those options, there is no effect.
        :return: best fit model of the current state of the FittingSequence class
        """
        param_class = self._updateManager.param_class(self._lens_temp)
        if bijective is False:
            lens_temp = param_class.update_lens_scaling(self._cosmo_temp, self._lens_temp, inverse=False)
            source_temp = param_class.image2source_plane(self._source_temp, lens_temp)
        else:
            lens_temp, source_temp = self._lens_temp, self._source_temp
        return lens_temp, source_temp, self._lens_light_temp, self._ps_temp, self._cosmo_temp

    @property
    def best_fit_likelihood(self):
        """
        returns the log likelihood of the best fit model of the current state of this class

        :return: log likelihood, float
        """
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo = self.best_fit(bijective=False)
        param_class = self._param_class
        likelihoodModule = self.likelihoodModule
        logL, _ = likelihoodModule.logL(param_class.kwargs2args(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                             kwargs_cosmo))
        return logL

    @property
    def _param_class(self):
        """

        :return: Param() class instance reflecting the current state of Fittingsequence
        """
        return self._updateManager.param_class(self._lens_temp)

    @property
    def likelihoodModule(self):
        """

        :return: Likelihood() class instance reflecting the current state of Fittingsequence
        """
        kwargs_model = self._updateManager.kwargs_model
        kwargs_likelihood = self._updateManager.kwargs_likelihood
        param_class = self._updateManager.param_class(self._lens_temp)
        likelihoodModule = LikelihoodModule(self.kwargs_data_joint, kwargs_model, param_class, **kwargs_likelihood)
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
        :return: MCMC samples, parameter names, logL distances of all samples
        """

        param_class = self._param_class
        # run PSO
        mcmc_class = Sampler(likelihoodModule=self.likelihoodModule)
        mean_start = param_class.kwargs2args(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                           self._cosmo_temp)
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
        elif sampler_type is 'EMCEE':
            n_walkers = num_param * walkerRatio
            samples = mcmc_class.mcmc_emcee(n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=self._mpi)
            dist = None
        else:
            raise ValueError('sampler_type %s not supported!' % sampler_type)
        return samples, param_list, dist

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

        param_class = self._param_class
        init_pos = param_class.kwargs2args(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                           self._cosmo_temp)
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

    def psf_iteration(self, num_iter=10, no_break=True, stacking_method='median', block_center_neighbour=0, keep_psf_error_map=True,
                 psf_symmetry=1, psf_iter_factor=1, verbose=True, compute_bands=None):
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
        #lens_temp = copy.deepcopy(lens_input)
        kwargs_model = self._updateManager.kwargs_model
        param_class = self._param_class
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
                                                               kwargs_model=kwargs_model)
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
        self._updateManager.update_fixed(self._lens_temp, self._source_temp, self._lens_light_temp, self._ps_temp,
                                         self._cosmo_temp, lens_add_fixed, source_add_fixed, lens_light_add_fixed,
                                         ps_add_fixed, cosmo_add_fixed, lens_remove_fixed, source_remove_fixed,
                                         lens_light_remove_fixed, ps_remove_fixed, cosmo_remove_fixed)
        self._updateManager.update_limits(change_source_lower_limit, change_source_upper_limit)
        return 0

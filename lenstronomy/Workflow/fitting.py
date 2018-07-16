__author__ = 'sibirrer'

import numpy as np

from lenstronomy.Sampling.mcmc import MCMCSampler
from lenstronomy.Sampling.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.parameters import Param, ParamUpdate


class Fitting(object):
    """
    class to find a good estimate of the parameter positions and uncertainties to run a (full) MCMC on
    """

    def __init__(self, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params):
        """

        :return:
        """
        self.multi_band_list = multi_band_list
        self.kwargs_model = kwargs_model
        self.kwargs_constraints = kwargs_constraints
        self.kwargs_likelihood = kwargs_likelihood

        if kwargs_model.get('lens_model_list', None) is not None:
            self._lens_init, self._lens_sigma, self._lens_fixed, self._lens_lower, self._lens_upper = kwargs_params['lens_model']
        else:
            self._lens_init, self._lens_sigma, self._lens_fixed, self._lens_lower, self._lens_upper = [], [], [], [], []
        if kwargs_model.get('source_light_model_list', None) is not None:
            self._source_init, self._source_sigma, self._source_fixed, self._source_lower, self._source_upper = kwargs_params['source_model']
        else:
            self._source_init, self._source_sigma, self._source_fixed, self._source_lower, self._source_upper = [], [], [], [], []
        if kwargs_model.get('lens_light_model_list', None) is not None:
            self._lens_light_init, self._lens_light_sigma, self._lens_light_fixed, self._lens_light_lower, self._lens_light_upper = kwargs_params['lens_light_model']
        else:
            self._lens_light_init, self._lens_light_sigma, self._lens_light_fixed, self._lens_light_lower, self._lens_light_upper = [], [], [], [], []
        if kwargs_model.get('point_source_model_list', None) is not None:
            self._ps_init, self._ps_sigma, self._ps_fixed, self._ps_lower, self._ps_upper = kwargs_params['point_source_model']
        else:
            self._ps_init, self._ps_sigma, self._ps_fixed, self._ps_lower, self._ps_upper = [], [], [], [], []
        if self.kwargs_likelihood.get('time_delay_likelihood', False) is True or self.kwargs_constraints.get('mass_scaling', False) is True:
            self._cosmo_init, self._cosmo_sigma, self._cosmo_fixed, self._cosmo_lower, self._cosmo_upper = kwargs_params['cosmography']
        else:
            self._cosmo_init, self._cosmo_sigma, self._cosmo_fixed, self._cosmo_lower, self._cosmo_upper = {}, {}, {}, {}, {}

        self._paramUpdate = ParamUpdate(self._lens_fixed, self._source_fixed, self._lens_light_fixed, self._ps_fixed, self._cosmo_fixed)

    def init_kwargs(self):
        return self._lens_init, self._source_init, self._lens_light_init, self._ps_init, self._cosmo_init

    def sigma_kwargs(self):
        return self._lens_sigma, self._source_sigma, self._lens_light_sigma, self._ps_sigma, self._cosmo_sigma

    def lower_kwargs(self):
        return self._lens_lower, self._source_lower, self._lens_light_lower, self._ps_lower, self._cosmo_lower

    def upper_kwargs(self):
        return self._lens_upper, self._source_upper, self._lens_light_upper, self._ps_upper, self._cosmo_upper

    def _run_pso(self, n_particles, n_iterations,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_ps, kwargs_mean_ps, kwargs_sigma_ps,
                 kwargs_fixed_cosmo, kwargs_mean_cosmo, kwargs_sigma_cosmo,
                 threadCount=1, mpi=False, print_key='Default', sigma_factor=1, compute_bool=None, fix_solver=False):
        kwargs_prior_lens = self._set_priors(kwargs_mean_lens, kwargs_sigma_lens)
        kwargs_prior_source = self._set_priors(kwargs_mean_source, kwargs_sigma_source)
        kwargs_prior_lens_light = self._set_priors(kwargs_mean_lens_light, kwargs_sigma_lens_light)
        kwargs_prior_ps = self._set_priors(kwargs_mean_ps, kwargs_sigma_ps)
        kwargs_prior_cosmo = kwargs_mean_cosmo.copy()
        kwargs_prior_cosmo.update(kwargs_sigma_cosmo)

        # initialise mcmc classes
        param_class = Param(self.kwargs_model, self.kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo,
                            kwargs_lens_init=kwargs_mean_lens, fix_lens_solver=fix_solver)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_ps, kwargs_prior_cosmo)
        lowerLimit = np.array(mean_start) - np.array(sigma_start)*sigma_factor
        upperLimit = np.array(mean_start) + np.array(sigma_start)*sigma_factor
        num_param, param_list = param_class.num_param()
        init_pos = param_class.setParams(kwargs_mean_lens, kwargs_mean_source,
                                         kwargs_mean_lens_light, kwargs_mean_ps, kwargs_mean_cosmo)
        # run PSO
        kwargs_fixed = [kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo]
        kwargs_lower = self.lower_kwargs()
        kwargs_upper = self.upper_kwargs()
        mcmc_class = MCMCSampler(self.multi_band_list, self.kwargs_model, self.kwargs_constraints,
                                 self.kwargs_likelihood, kwargs_fixed, kwargs_lower, kwargs_upper,
                                 kwargs_lens_init=kwargs_mean_lens, compute_bool=compute_bool, fix_solver=fix_solver)

        lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain = mcmc_class.pso(n_particles,
                                                                                                       n_iterations,
                                                                                                       lowerLimit,
                                                                                                       upperLimit,
                                                                                                       init_pos=init_pos,
                                                                                                       threadCount=threadCount,
                                                                                                       mpi=mpi,
                                                                                                       print_key=print_key)
        return lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param_list

    def _mcmc_run(self, n_burn, n_run, walkerRatio,
                  kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                  kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                  kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                  kwargs_fixed_ps, kwargs_mean_ps, kwargs_sigma_ps,
                  kwargs_fixed_cosmo, kwargs_mean_cosmo, kwargs_sigma_cosmo,
                  threadCount=1, mpi=False, init_samples=None, sigma_factor=1, compute_bool=None, fix_solver=False):

        kwargs_prior_lens = self._set_priors(kwargs_mean_lens, kwargs_sigma_lens)
        kwargs_prior_source = self._set_priors(kwargs_mean_source, kwargs_sigma_source)
        kwargs_prior_lens_light = self._set_priors(kwargs_mean_lens_light, kwargs_sigma_lens_light)
        kwargs_prior_ps = self._set_priors(kwargs_mean_ps, kwargs_sigma_ps)
        kwargs_prior_cosmo = kwargs_mean_cosmo.copy()
        kwargs_prior_cosmo.update(kwargs_sigma_cosmo)
        # initialise mcmc classes

        param_class = Param(self.kwargs_model, self.kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo,
                            kwargs_lens_init=kwargs_mean_lens, fix_lens_solver=fix_solver)
        kwargs_fixed = [kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps,
                        kwargs_fixed_cosmo]
        kwargs_lower = self.lower_kwargs()
        kwargs_upper = self.upper_kwargs()

        mcmc_class = MCMCSampler(self.multi_band_list, self.kwargs_model, self.kwargs_constraints,
                                 self.kwargs_likelihood, kwargs_fixed, kwargs_lower, kwargs_upper,
                                 kwargs_lens_init=kwargs_mean_lens, compute_bool=compute_bool, fix_solver=fix_solver)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_ps, kwargs_prior_cosmo)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_samples is None:
            initpos = ReusePositionGenerator(init_samples)
        else:
            initpos = None
        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, np.array(sigma_start)*sigma_factor, threadCount=threadCount,
                                           mpi=mpi, init_pos=initpos)
        return samples, param_list, dist

    def pso_run(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo,
                n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1,
                compute_bool=None, fix_lens=False, fix_source=False, fix_lens_light=False, fix_point_source=False,
                fix_cosmo=False, print_key='find_model'):
        """
        finds lens light and lens model combined fit
        :return: constraints of lens model
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo = self._paramUpdate.update_fixed_simple(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo, fix_lens=fix_lens, fix_source=fix_source,
            fix_lens_light=fix_lens_light, fix_point_source=fix_point_source, fixed_cosmo=fix_cosmo)
        kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_ps_sigma, kwargs_cosmo_sigma = self.sigma_kwargs()
        lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_ps, kwargs_ps, kwargs_ps_sigma,
            kwargs_fixed_cosmo, kwargs_cosmo, kwargs_cosmo_sigma,
            threadCount=threadCount, mpi=mpi, print_key=print_key, sigma_factor=sigma_factor, compute_bool=compute_bool,
            fix_solver=fix_lens)
        return lens_result, source_result, lens_light_result, ps_result, cosmo_result, chain, param_list

    def mcmc_run(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=False, init_samples=None, sigma_factor=1,
                 compute_bool=None, fix_lens=False, fix_source=False, fix_lens_light=False,
                 fix_point_source=False):
        """
        MCMC
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_fixed_cosmo = self._paramUpdate.update_fixed_simple(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo, fix_lens=fix_lens, fix_source=fix_source,
            fix_lens_light=fix_lens_light, fix_point_source=fix_point_source)
        kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_ps_sigma, kwargs_cosmo_sigma = self.sigma_kwargs()
        samples, param_list, dist = self._mcmc_run(
            n_burn, n_run, walkerRatio,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_ps, kwargs_ps, kwargs_ps_sigma,
            kwargs_fixed_cosmo, kwargs_cosmo, kwargs_cosmo_sigma,
            threadCount=threadCount, mpi=mpi, init_samples=init_samples, sigma_factor=sigma_factor,
            compute_bool=compute_bool, fix_solver=fix_lens)
        return samples, param_list, dist

    def _set_priors(self, mean_list_kwargs, sigma_list_kwargs):
        """

        :param mean_list_kwargs:
        :param sigma_list_kwargs:
        :return:
        """
        prior_list = []
        for k in range(len(mean_list_kwargs)):
            prior_k = mean_list_kwargs[k].copy()
            prior_k.update(sigma_list_kwargs[k])
            prior_list.append(prior_k)
        return prior_list

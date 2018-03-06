__author__ = 'sibirrer'

import numpy as np

from lenstronomy.MCMC.mcmc import MCMCSampler
from lenstronomy.MCMC.reinitialize import ReusePositionGenerator
from lenstronomy.Workflow.parameters import Param, ParamUpdate


class Fitting(object):
    """
    class to find a good estimate of the parameter positions and uncertainties to run a (full) MCMC on
    """

    def __init__(self, multi_band_list, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_fixed, kwargs_lower,
                 kwargs_upper):
        """

        :return:
        """
        self.multi_band_list = multi_band_list
        self.kwargs_model = kwargs_model
        self.kwargs_constraints = kwargs_constraints
        self.kwargs_likelihood = kwargs_likelihood
        self.kwargs_lower = kwargs_lower
        self.kwargs_upper = kwargs_upper
        lens_fix, source_fix, lens_light_fix, ps_fix = kwargs_fixed
        self._paramUpdate = ParamUpdate(lens_fix, source_fix, lens_light_fix, ps_fix)

    def _run_pso(self, n_particles, n_iterations,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_ps, kwargs_mean_ps, kwargs_sigma_ps,
                 threadCount=1, mpi=False, print_key='Default', sigma_factor=1, compute_bool=None):

        kwargs_prior_lens = self._set_priors(kwargs_mean_lens, kwargs_sigma_lens)
        kwargs_prior_source = self._set_priors(kwargs_mean_source, kwargs_sigma_source)
        kwargs_prior_lens_light = self._set_priors(kwargs_mean_lens_light, kwargs_sigma_lens_light)
        kwargs_prior_ps = self._set_priors(kwargs_mean_ps, kwargs_sigma_ps)

        # initialise mcmc classes
        param_class = Param(self.kwargs_model, self.kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_ps, kwargs_lens_init=kwargs_mean_lens)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_ps)
        lowerLimit = np.array(mean_start) - np.array(sigma_start)*sigma_factor
        upperLimit = np.array(mean_start) + np.array(sigma_start)*sigma_factor
        num_param, param_list = param_class.num_param()
        init_pos = param_class.setParams(kwargs_mean_lens, kwargs_mean_source,
                                         kwargs_mean_lens_light, kwargs_mean_ps)
        # run PSO
        kwargs_fixed = [kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps]
        mcmc_class = MCMCSampler(self.multi_band_list, self.kwargs_model, self.kwargs_constraints,
                                 self.kwargs_likelihood, kwargs_fixed, self.kwargs_lower, self.kwargs_upper,
                                 kwargs_lens_init=kwargs_mean_lens, compute_bool=compute_bool)
        lens_result, source_result, lens_light_result, else_result, chain = mcmc_class.pso(n_particles,
                                                                                                       n_iterations,
                                                                                                       lowerLimit,
                                                                                                       upperLimit,
                                                                                                       init_pos=init_pos,
                                                                                                       threadCount=threadCount,
                                                                                                       mpi=mpi,
                                                                                                       print_key=print_key)
        return lens_result, source_result, lens_light_result, else_result, chain, param_list

    def _mcmc_run(self, n_burn, n_run, walkerRatio,
                 kwargs_fixed_lens, kwargs_mean_lens, kwargs_sigma_lens,
                 kwargs_fixed_source, kwargs_mean_source, kwargs_sigma_source,
                 kwargs_fixed_lens_light, kwargs_mean_lens_light, kwargs_sigma_lens_light,
                 kwargs_fixed_else, kwargs_mean_ps, kwargs_sigma_ps,
                 threadCount=1, mpi=False, init_samples=None, sigma_factor=1, compute_bool=None):

        kwargs_prior_lens = self._set_priors(kwargs_mean_lens, kwargs_sigma_lens)
        kwargs_prior_source = self._set_priors(kwargs_mean_source, kwargs_sigma_source)
        kwargs_prior_lens_light = self._set_priors(kwargs_mean_lens_light, kwargs_sigma_lens_light)
        kwargs_prior_ps = self._set_priors(kwargs_mean_ps, kwargs_sigma_ps)
        # initialise mcmc classes

        param_class = Param(self.kwargs_model, self.kwargs_constraints, kwargs_fixed_lens, kwargs_fixed_source,
                            kwargs_fixed_lens_light, kwargs_fixed_else, kwargs_lens_init=kwargs_mean_lens)
        kwargs_fixed = kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else

        mcmc_class = MCMCSampler(self.multi_band_list, self.kwargs_model, self.kwargs_constraints,
                                 self.kwargs_likelihood, kwargs_fixed, self.kwargs_lower, self.kwargs_upper,
                                 kwargs_lens_init=kwargs_mean_lens, compute_bool=compute_bool)
        mean_start, sigma_start = param_class.param_init(kwargs_prior_lens, kwargs_prior_source,
                                                         kwargs_prior_lens_light, kwargs_prior_ps)
        num_param, param_list = param_class.num_param()
        # run MCMC
        if not init_samples is None:
            initpos = ReusePositionGenerator(init_samples)
        else:
            initpos = None

        samples, dist = mcmc_class.mcmc_CH(walkerRatio, n_run, n_burn, mean_start, np.array(sigma_start)*sigma_factor, threadCount=threadCount,
                                           mpi=mpi, init_pos=initpos)
        return samples, param_list, dist

    def pso_run(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_ps_sigma,
                n_particles, n_iterations, mpi=False, threadCount=1, sigma_factor=1, gamma_fixed=False,
                compute_bool=None, fix_lens=False, fix_source=False, fix_lens_light=False, fix_point_source=False,
                print_key='find_model'):
        """
        finds lens light and lens model combined fit
        :return: constraints of lens model
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps = self._paramUpdate.update_fixed_simple(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, fix_lens=fix_lens, fix_source=fix_source,
            fix_lens_light=fix_lens_light, fix_point_source=fix_point_source, gamma_fixed=gamma_fixed)
        lens_result, source_result, lens_light_result, ps_result, chain, param_list = self._run_pso(
            n_particles, n_iterations,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_ps, kwargs_ps, kwargs_ps_sigma,
            threadCount=threadCount, mpi=mpi, print_key=print_key, sigma_factor=sigma_factor, compute_bool=compute_bool)
        return lens_result, source_result, lens_light_result, ps_result, chain, param_list

    def mcmc_run(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                 kwargs_lens_sigma, kwargs_source_sigma, kwargs_lens_light_sigma, kwargs_ps_sigma,
                 n_burn, n_run, walkerRatio, threadCount=1, mpi=False, init_samples=None, sigma_factor=1,
                 gamma_fixed=False, compute_bool=None, fix_lens=False, fix_source=False, fix_lens_light=False,
                 fix_point_source=False):
        """
        MCMC
        """
        kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_ps = self._paramUpdate.update_fixed_simple(
            kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, fix_lens=fix_lens, fix_source=fix_source,
            fix_lens_light=fix_lens_light, fix_point_source=fix_point_source, gamma_fixed=gamma_fixed)

        samples, param_list, dist = self._mcmc_run(
            n_burn, n_run, walkerRatio,
            kwargs_fixed_lens, kwargs_lens, kwargs_lens_sigma,
            kwargs_fixed_source, kwargs_source, kwargs_source_sigma,
            kwargs_fixed_lens_light, kwargs_lens_light, kwargs_lens_light_sigma,
            kwargs_fixed_ps, kwargs_ps, kwargs_ps_sigma,
            threadCount=threadCount, mpi=mpi, init_samples=init_samples, sigma_factor=sigma_factor, compute_bool=compute_bool)
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

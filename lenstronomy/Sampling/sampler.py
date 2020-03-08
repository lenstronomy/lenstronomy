__author__ = ['sibirrer', 'ajshajib']

import time
import sys

import numpy as np
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Util import sampling_util
import emcee
from schwimmbad import choose_pool


class Sampler(object):
    """
    class which executes the different sampling  methods
    Available are: MCMC with emcee and comsoHammer and a Particle Swarm Optimizer.
    This are examples and depending on your problem, you might find other/better solutions.
    Feel free to sample with your convenient sampler!

    """
    def __init__(self, likelihoodModule):
        """

        :param likelihoodModule: instance of LikelihoodModule class
        """
        self.chain = likelihoodModule
        self.lower_limit, self.upper_limit = self.chain.param_limits

    def pso(self, n_particles, n_iterations, lower_start=None, upper_start=None,
            threadCount=1, init_pos=None, mpi=False, print_key='PSO'):
        """
        Return the best fit for the lens model on catalogue basis with
        particle swarm optimizer.
        """
        if lower_start is None or upper_start is None:
            lower_start, upper_start = np.array(self.lower_limit), np.array(self.upper_limit)
            print("PSO initialises its particles with default values")
        else:
            lower_start = np.maximum(lower_start, self.lower_limit)
            upper_start = np.minimum(upper_start, self.upper_limit)

        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        if mpi is True and pool.is_master():
            print('MPI option chosen for PSO.')

        pso = ParticleSwarmOptimizer(self.chain.logL,
                                     lower_start, upper_start, n_particles,
                                     pool=pool)

        if init_pos is None:
            init_pos = (upper_start - lower_start) / 2 + lower_start

        pso.set_global_best(init_pos, [0]*len(init_pos),
                            self.chain.logL(init_pos))

        if pool.is_master():
            print('Computing the %s ...' % print_key)

        time_start = time.time()

        result, [chi2_list, pos_list, vel_list] = pso.optimize(n_iterations)

        if pool.is_master():
            kwargs_return = self.chain.param.args2kwargs(result)
            print(pso.global_best.fitness * 2 / (max(
                self.chain.effective_num_data_points(**kwargs_return), 1)), 'reduced X^2 of best position')
            print(pso.global_best.fitness, 'logL')
            print(self.chain.effective_num_data_points(**kwargs_return), 'effective number of data points')
            print(kwargs_return.get('kwargs_lens', None), 'lens result')
            print(kwargs_return.get('kwargs_source', None), 'source result')
            print(kwargs_return.get('kwargs_lens_light', None), 'lens light result')
            print(kwargs_return.get('kwargs_ps', None), 'point source result')
            print(kwargs_return.get('kwargs_special', None), 'special param result')
            time_end = time.time()
            print(time_end - time_start, 'time used for ', print_key)
            print('===================')
        return result, [chi2_list, pos_list, vel_list, []]

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=False, progress=False, threadCount=1):
        """
        Run MCMC with emcee.

        :param n_walkers:
        :type n_walkers:
        :param n_run:
        :type n_run:
        :param n_burn:
        :type n_burn:
        :param mean_start:
        :type mean_start:
        :param sigma_start:
        :type sigma_start:
        :param mpi:
        :type mpi:
        :param progress:
        :type progress:
        :param threadCount:
        :type threadCount:
        :return:
        :rtype:
        """
        num_param, _ = self.chain.param.num_param()
        p0 = sampling_util.sample_ball(mean_start, sigma_start, n_walkers)
        time_start = time.time()

        pool = choose_pool(mpi=mpi, processes=threadCount)

        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.logL,
                                        pool=pool)

        sampler.run_mcmc(p0, n_burn + n_run, progress=progress)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=1)
        if pool.is_master():
            print('Computing the MCMC...')
            print('Number of walkers = ', n_walkers)
            print('Burn-in iterations: ', n_burn)
            print('Sampling iterations:', n_run)
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        return flat_samples, dist

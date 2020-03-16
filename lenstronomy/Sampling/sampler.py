__author__ = ['sibirrer', 'ajshajib', 'dgilman']

import time

import numpy as np
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Util import sampling_util
import emcee
import schwimmbad
from scipy.optimize import minimize


def choose_pool(mpi=False, processes=1, **kwargs):
    """
    Extends the capabilities of the schwimmbad.choose_pool method.
    
    It handles the `use_dill` parameters in kwargs, that would otherwise raise an error when processes > 1.
    Any thread in the returned multiprocessing pool (e.g. processes > 1) also default
    

    Docstring from schwimmbad:

    mpi : bool, optional
        Use the MPI processing pool, :class:`~schwimmbad.mpi.MPIPool`. By
        default, ``False``, will use the :class:`~schwimmbad.serial.SerialPool`.
    processes : int, optional
        Use the multiprocessing pool,
        :class:`~schwimmbad.multiprocessing.MultiPool`, with this number of
        processes. By default, ``processes=1``, will use the
        :class:`~schwimmbad.serial.SerialPool`.
    **kwargs
        Any additional kwargs are passed in to the pool class initializer
        selected by the arguments.
    """
    if processes == 1 or mpi:
        pool = schwimmbad.choose_pool(mpi=mpi, processes=1, **kwargs)
        is_master = pool.is_master()
    else:
        if 'use_dill' in kwargs:
            # schwimmbad MultiPool does not support dill so we remove this option from the kwargs
            _ = kwargs.pop('use_dill')
        pool = schwimmbad.choose_pool(mpi=False, processes=processes, **kwargs)
        # this MultiPool has no is_master() attribute like the SerialPool and MpiPool
        # all threads will then be 'master'.
        is_master = True
    return pool, is_master


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

    def simplex(self, init_pos, n_iterations, method, print_key='SIMPLEX'):
        """

        :param init_pos: starting point for the optimization
        :param n_iterations: maximum number of iterations
        :param method: the optimization method, default is 'Nelder-Mead'
        returns the best fit for the lens model using the optimization routine specified by method
        """
        print('Performing the optimization using algorithm:', method)
        time_start = time.time()

        #negativelogL = lambda x: -1 * self.chain.logL(x)

        result = minimize(self.chain.negativelogL, x0=init_pos, method=method,
                          options={'maxiter': n_iterations, 'disp': True})
        logL = self.chain.logL(result['x'])
        kwargs_return = self.chain.param.args2kwargs(result['x'])
        print(-logL * 2 / (max(self.chain.effective_num_data_points(**kwargs_return), 1)),
              'reduced X^2 of best position')
        print(logL, 'logL')
        print(self.chain.effective_num_data_points(**kwargs_return), 'effective number of data points')
        print(kwargs_return.get('kwargs_lens', None), 'lens result')
        print(kwargs_return.get('kwargs_source', None), 'source result')
        print(kwargs_return.get('kwargs_lens_light', None), 'lens light result')
        print(kwargs_return.get('kwargs_ps', None), 'point source result')
        print(kwargs_return.get('kwargs_special', None), 'special param result')
        time_end = time.time()
        print(time_end - time_start, 'time used for ', print_key)
        print('===================')

        return result['x']

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

        pool, is_master = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)
        
        if mpi is True and is_master:
            print('MPI option chosen for PSO.')

        pso = ParticleSwarmOptimizer(self.chain.logL,
                                     lower_start, upper_start, n_particles,
                                     pool=pool)

        if init_pos is None:
            init_pos = (upper_start - lower_start) / 2 + lower_start

        pso.set_global_best(init_pos, [0]*len(init_pos),
                            self.chain.logL(init_pos))

        if is_master:
            print('Computing the %s ...' % print_key)

        time_start = time.time()

        result, [chi2_list, pos_list, vel_list] = pso.optimize(n_iterations)

        if is_master:
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

        pool, is_master = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.logL,
                                        pool=pool)

        sampler.run_mcmc(p0, n_burn + n_run, progress=progress)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=1)
        if is_master:
            print('Computing the MCMC...')
            print('Number of walkers = ', n_walkers)
            print('Burn-in iterations: ', n_burn)
            print('Sampling iterations:', n_run)
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        return flat_samples, dist

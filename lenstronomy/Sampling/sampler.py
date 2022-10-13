__author__ = ['sibirrer', 'ajshajib', 'dgilman', 'nataliehogg']

import time

import numpy as np
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from lenstronomy.Util import sampling_util
from lenstronomy.Sampling.Pool.pool import choose_pool
from scipy.optimize import minimize

__all__ = ['Sampler']


class Sampler(object):
    """
    class which executes the different sampling  methods
    Available are: MCMC with emcee a Particle Swarm Optimizer.
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
        :return: the best fit for the lens model using the optimization routine specified by method
        """
        print('Performing the optimization using algorithm:', method)
        time_start = time.time()

        result = minimize(self.chain.negativelogL, x0=init_pos, method=method,
                          options={'maxiter': n_iterations, 'disp': True})
        logL = self.chain.logL(result['x'])
        kwargs_return = self.chain.param.args2kwargs(result['x'])
        print(-logL * 2 / (max(self.chain.effective_num_data_points(**kwargs_return), 1)),
              'reduced X^2 of best position')
        print(logL, 'log likelihood')
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

        :param n_particles: number of particles in the sampling process
        :param n_iterations: number of iterations of the swarm
        :param lower_start: numpy array, lower end parameter of the values of the starting particles
        :param upper_start: numpy array, upper end parameter of the values of the starting particles
        :param threadCount: number of threads in the computation (only applied if mpi=False)
        :param init_pos: numpy array, position of the initial best guess model
        :param mpi: bool, if True, makes instance of MPIPool to allow for MPI execution
        :param print_key: string, prints the process name in the progress bar (optional)
        :return: kwargs_result (of best fit), [lnlikelihood of samples, positions of samples, velocity of samples])
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

        result, [log_likelihood_list, pos_list, vel_list] = pso.optimize(n_iterations)

        if pool.is_master():
            kwargs_return = self.chain.param.args2kwargs(result)
            print(pso.global_best.fitness * 2 / (max(
                self.chain.effective_num_data_points(**kwargs_return), 1)), 'reduced X^2 of best position')
            print(pso.global_best.fitness, 'log likelihood')
            print(self.chain.effective_num_data_points(**kwargs_return), 'effective number of data points')
            print(kwargs_return.get('kwargs_lens', None), 'lens result')
            print(kwargs_return.get('kwargs_source', None), 'source result')
            print(kwargs_return.get('kwargs_lens_light', None), 'lens light result')
            print(kwargs_return.get('kwargs_ps', None), 'point source result')
            print(kwargs_return.get('kwargs_special', None), 'special param result')
            time_end = time.time()
            print(time_end - time_start, 'time used for ', print_key)
            print('===================')
        return result, [log_likelihood_list, pos_list, vel_list]

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start,
                   mpi=False, progress=False, threadCount=1,
                   initpos=None, backend_filename=None, start_from_backend=False):
        """
        Run MCMC with emcee.
        For details, please have a look at the documentation of the emcee packager.

        :param n_walkers: number of walkers in the emcee process
        :type n_walkers: integer
        :param n_run: number of sampling (after burn-in) of the emcee
        :type n_run: integer
        :param n_burn: number of burn-in iterations (those will not be saved in the output sample)
        :type n_burn: integer
        :param mean_start: mean of the parameter position of the initialising sample
        :type mean_start: numpy array of length the number of parameters
        :param sigma_start: spread of the parameter values (uncorrelated in each dimension) of the initialising sample
        :type sigma_start: numpy array of length the number of parameters
        :param mpi: if True, initializes an MPIPool to allow for MPI execution of the sampler
        :type mpi: bool
        :param progress: if True, prints the progress bar
        :type progress: bool
        :param threadCount: number of threats in multi-processing (not applicable for MPI)
        :type threadCount: integer
        :param initpos: initial walker position to start sampling (optional)
        :type initpos: numpy array of size num param x num walkser
        :param backend_filename: name of the HDF5 file where sampling state is saved (through emcee backend engine)
        :type backend_filename: string
        :param start_from_backend: if True, start from the state saved in `backup_filename`.
         Otherwise, create a new backup file with name `backup_filename` (any already existing file is overwritten!).
        :type start_from_backend: bool
        :return: samples, ln likelihood value of samples
        :rtype: numpy 2d array, numpy 1d array
        """
        import emcee

        num_param, _ = self.chain.param.num_param()
        if initpos is None:
            initpos = sampling_util.sample_ball_truncated(mean_start, sigma_start, self.lower_limit, self.upper_limit,
                                                          size=n_walkers)

        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        if backend_filename is not None:
            backend = emcee.backends.HDFBackend(backend_filename, name="lenstronomy_mcmc_emcee")
            if pool.is_master():
                print("Warning: All samples (including burn-in) will be saved in backup file '{}'.".format(backend_filename))
            if start_from_backend:
                initpos = None
                n_run_eff = n_run
            else:
                n_run_eff = n_burn + n_run
                backend.reset(n_walkers, num_param)
                if pool.is_master():
                    print("Warning: backup file '{}' has been reset!".format(backend_filename))
        else:
            backend = None
            n_run_eff = n_burn + n_run

        time_start = time.time()

        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.logL, pool=pool, backend=backend)

        sampler.run_mcmc(initpos, n_run_eff, progress=progress)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=1)
        if pool.is_master():
            print('Computing the MCMC...')
            print('Number of walkers = ', n_walkers)
            print('Burn-in iterations: ', n_burn)
            print('Sampling iterations (in current run):', n_run_eff)
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        return flat_samples, dist

    def mcmc_zeus(self, n_walkers, n_run, n_burn, mean_start, sigma_start,
                  mpi=False, threadCount=1,
                  progress=False, initpos=None, backend_filename=None,
                  moves=None, tune=True, tolerance=0.05, patience=5,
                  maxsteps=10000, mu=1.0, maxiter=10000, pool=None,
                  vectorize=False, blobs_dtype=None, verbose=True,
                  check_walkers=True, shuffle_ensemble=True, light_mode=False):

        """
        Lightning fast MCMC with zeus: https://github.com/minaskar/zeus

        For the full list of arguments for the EnsembleSampler, see see `the zeus docs <https://zeus-mcmc.readthedocs.io/en/latest/api/sampler.html>`_.

        If you use the zeus sampler, you should cite the following papers: 2105.03468, 2002.06212.

        :param n_walkers: number of walkers per parameter
        :type n_walkers: integer
        :param n_run: number of sampling steps
        :type n_run: integer
        :param n_burn: number of burn-in steps
        :type n_burn: integer
        :param mean_start: mean of the parameter position of the initialising sample
        :type mean_start: numpy array of length the number of parameters
        :param sigma_start: spread of the parameter values (uncorrelated in each dimension) of the initialising sample
        :type sigma_start: numpy array of length the number of parameters
        :param mpi: if True, initializes an MPIPool to allow for MPI execution of the sampler
        :type mpi: bool
        :param progress:
        :type progress: bool
        :param initpos: initial walker position to start sampling (optional)
        :type initpos: numpy array of size num param x num walkser
        :param backend_filename: name of the HDF5 file where sampling state is saved (through zeus callback function)
        :type backend_filename: string
        :return: samples, ln likelihood value of samples
        :rtype: numpy 2d array, numpy 1d array
        """
        import zeus

        print('Using zeus to perform the MCMC.')

        num_param, _ = self.chain.param.num_param()

        if initpos is None:
            initpos = sampling_util.sample_ball_truncated(mean_start, sigma_start, self.lower_limit, self.upper_limit,
                                                          size=n_walkers)

        if backend_filename is not None:
            backend = zeus.callbacks.SaveProgressCallback(filename= backend_filename, ncheck = 1)
            n_run_eff = n_burn + n_run
        else:
            backend = None
            n_run_eff = n_burn + n_run

        pool = choose_pool(mpi=mpi, processes=threadCount, use_dill=True)

        sampler = zeus.EnsembleSampler(nwalkers=n_walkers, ndim=num_param, logprob_fn=self.chain.logL,
                                       moves=moves, tune=tune, tolerance=tolerance, patience=patience,
                                       maxsteps=maxsteps, mu=mu, maxiter=maxiter, pool=pool, vectorize=vectorize,
                                       blobs_dtype=blobs_dtype, verbose=verbose, check_walkers=check_walkers,
                                       shuffle_ensemble=shuffle_ensemble, light_mode=light_mode)

        sampler.run_mcmc(initpos, n_run_eff, progress=progress, callbacks=backend)

        flat_samples = sampler.get_chain(flat=True, thin=1, discard=n_burn)

        dist = sampler.get_log_prob(flat=True, thin=1, discard=n_burn)

        return flat_samples, dist

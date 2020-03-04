__author__ = 'sibirrer'


import os
import shutil
import tempfile
import time
import sys

from cosmoHammer import CosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer.util import InMemoryStorageUtil
from cosmoHammer.util import MpiUtil


import numpy as np

from lenstronomy.Util import sampling_util
import emcee
from schwimmbad import MPIPool
from multiprocessing import Pool
#from multiprocess import Pool


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

    def pso(self, n_particles, n_iterations, lower_start=None, upper_start=None, threadCount=1, init_pos=None,
            mpi=False, print_key='PSO', mpipso=True):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
        if lower_start is None or upper_start is None:
            lower_start, upper_start = np.array(self.lower_limit), np.array(self.upper_limit)
            print("PSO initialises its particles with default values")
        else:
            lower_start = np.maximum(lower_start, self.lower_limit)
            upper_start = np.minimum(upper_start, self.upper_limit)
        if mpi is True:
            if mpipso is True:
                from mpipso.mpipso import MpiParticleSwarmOptimizer
            else:
                from cosmoHammer import MpiParticleSwarmOptimizer
            pso = MpiParticleSwarmOptimizer(self.chain.likelihood_derivative, lower_start, upper_start, n_particles, threads=1)
            if pso.isMaster():
                print('MPI option chosen')
        else:
            if mpipso is True:
                from mpipso.pso import ParticleSwarmOptimizer
            else:
                from cosmoHammer import ParticleSwarmOptimizer
            pso = ParticleSwarmOptimizer(self.chain.likelihood_derivative, lower_start, upper_start, n_particles, threads=threadCount)
        if init_pos is None:
            init_pos = (upper_start - lower_start) / 2 + lower_start
        if not init_pos is None:
            pso.gbest.position = init_pos
            pso.gbest.velocity = [0]*len(init_pos)
            pso.gbest.fitness = self.chain.likelihood(init_pos)
        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()
        if pso.isMaster():
            print('Computing the %s ...' % print_key)
        num_iter = 0
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness*2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            num_iter += 1
            if pso.isMaster():
                if num_iter % 10 == 0:
                    print(num_iter)
        if not mpi:
            result = pso.gbest.position
        else:
            result = MpiUtil.mpiBCast(pso.gbest.position)

        if mpi is True and not pso.isMaster():
            pass
        else:
            kwargs_return = self.chain.param.args2kwargs(result)
            print(pso.gbest.fitness * 2 / (max(self.chain.effectiv_num_data_points(**kwargs_return), 1)), 'reduced X^2 of best position')
            print(pso.gbest.fitness, 'logL')
            print(self.chain.effectiv_num_data_points(**kwargs_return), 'effective number of data points')
            print(kwargs_return.get('kwargs_lens', None), 'lens result')
            print(kwargs_return.get('kwargs_source', None), 'source result')
            print(kwargs_return.get('kwargs_lens_light', None), 'lens light result')
            print(kwargs_return.get('kwargs_ps', None), 'point source result')
            print(kwargs_return.get('kwargs_special', None), 'special param result')
            time_end = time.time()
            print(time_end - time_start, 'time used for ', print_key)
            print('===================')
        return result, [X2_list, pos_list, vel_list, []]

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start, mpi=False, progress=False, threadCount=1):
        numParam, _ = self.chain.param.num_param()
        p0 = sampling_util.sample_ball(mean_start, sigma_start, n_walkers)
        time_start = time.time()
        if mpi is True:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            is_master_pool = pool.is_master()
            sampler = emcee.EnsembleSampler(n_walkers, numParam, self.chain.logL, pool=pool)
        else:
            is_master_pool = True
            if threadCount > 1 :
                pool = Pool(processes=threadCount)
            else :
                pool = None
            sampler = emcee.EnsembleSampler(n_walkers, numParam, self.chain.likelihood, pool=pool)

        sampler.run_mcmc(p0, n_burn + n_run, progress=progress)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        dist = sampler.get_log_prob(flat=True, discard=n_burn, thin=1)
        if is_master_pool:
            print('Computing the MCMC...')
            print('Number of walkers = ', n_walkers)
            print('Burn-in iterations: ', n_burn)
            print('Sampling iterations:', n_run)
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        return flat_samples, dist

    def mcmc_CH(self, walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi=False):
        """
        runs mcmc on the parameter space given parameter bounds with CosmoHammerSampler
        returns the chain
        """
        lowerLimit, upperLimit = self.lower_limit, self.upper_limit

        mean_start = np.maximum(lowerLimit, mean_start)
        mean_start = np.minimum(upperLimit, mean_start)

        low_start = mean_start - sigma_start
        high_start = mean_start + sigma_start
        #low_start = np.maximum(lowerLimit, low_start)
        #high_start = np.minimum(upperLimit, high_start)
        #sigma_start = (high_start - low_start) / 2
        #mean_start = (high_start + low_start) / 2
        params = np.array([mean_start, lowerLimit, upperLimit, sigma_start]).T

        chain = LikelihoodComputationChain(
            min=lowerLimit,
            max=upperLimit)

        temp_dir = tempfile.mkdtemp("Hammer")
        file_prefix = os.path.join(temp_dir, "logs")
        #file_prefix = "./lenstronomy_debug"
        # chain.addCoreModule(CambCoreModule())
        chain.addLikelihoodModule(self.chain)
        chain.setup()

        store = InMemoryStorageUtil()
        #store = None
        if mpi is True:
            sampler = MpiCosmoHammerSampler(
            params=params,
            likelihoodComputationChain=chain,
            filePrefix=file_prefix,
            walkersRatio=walkerRatio,
            burninIterations=n_burn,
            sampleIterations=n_run,
            threadCount=1,
            initPositionGenerator=init_pos,
            storageUtil=store)
        else:
            sampler = CosmoHammerSampler(
                params=params,
                likelihoodComputationChain=chain,
                filePrefix=file_prefix,
                walkersRatio=walkerRatio,
                burninIterations=n_burn,
                sampleIterations=n_run,
                threadCount=threadCount,
                initPositionGenerator=init_pos,
                storageUtil=store)
        time_start = time.time()
        if sampler.isMaster():
            print('Computing the MCMC...')
            print('Number of walkers = ', len(mean_start)*walkerRatio)
            print('Burn-in iterations: ', n_burn)
            print('Sampling iterations:', n_run)
        sampler.startSampling()
        if sampler.isMaster():
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        # if sampler._sampler.pool is not None:
        #     sampler._sampler.pool.close()
        try:
            shutil.rmtree(temp_dir)
        except Exception as ex:
            print(ex, 'shutil.rmtree did not work')
            pass

        return store.samples, store.prob

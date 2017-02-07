__author__ = 'sibirrer'

import os
import shutil
import tempfile
import time

import emcee
import numpy as np
from cosmoHammer import CosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain
from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import MpiParticleSwarmOptimizer
from cosmoHammer import ParticleSwarmOptimizer
from cosmoHammer.util import InMemoryStorageUtil
from cosmoHammer.util import MpiUtil
from lenstronomy.MCMC.mcmc_chains import MCMC_chain, MCMC_multiband_chain
from lenstronomy.Workflow.parameters import Param


class MCMC_sampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else):
        """
        initialise the classes of the chain and for parameter options
        """
        if kwargs_options.get('multiBand', False):
            self.chain = MCMC_multiband_chain(kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
        else:
            self.chain = MCMC_chain(kwargs_data, kwargs_psf, kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)
        self.param = Param(kwargs_options, kwargs_fixed_lens, kwargs_fixed_source, kwargs_fixed_lens_light, kwargs_fixed_else)

    def pso(self, n_particles, n_iterations, lowerLimit=None, upperLimit=None, threadCount=1, init_pos=None, print_positions=False, mpi_monch=False, print_key='default'):
        """
        returns the best fit for the lense model on catalogue basis with particle swarm optimizer
        """
        if lowerLimit is None or upperLimit is None:
            lowerLimit, upperLimit = self.param.param_bounds()
            print("PSO initialises its particles with default values")
        if mpi_monch is True:
            pso = MpiParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=1)
        else:
            pso = ParticleSwarmOptimizer(self.chain, lowerLimit, upperLimit, n_particles, threads=threadCount)
        if not init_pos is None:
            pso.gbest.position = init_pos
            pso.gbest.velocity = [0]*len(init_pos)
            pso.gbest.fitness, _ = self.chain.likelihood(init_pos)
        X2_list = []
        vel_list = []
        pos_list = []
        time_start = time.time()
        if pso.isMaster():
            print('Computing the ', print_key, '...')
        num_iter = 0
        for swarm in pso.sample(n_iterations):
            X2_list.append(pso.gbest.fitness*2)
            vel_list.append(pso.gbest.velocity)
            pos_list.append(pso.gbest.position)
            num_iter += 1
            if pso.isMaster():
                if num_iter % 10 == 0:
                    print(num_iter)
        if not mpi_monch:
            result = pso.gbest.position
        else:
            result = MpiUtil.mpiBCast(pso.gbest.position)
        lens_dict, source_dict, lens_light_dict, else_dict = self.param.get_all_params(result)
        if (pso.isMaster() and mpi_monch is True) or self.chain.sampling_option == 'X2_catalogue':
            print(pso.gbest.fitness*2/(self.chain.numData_points()), 'reduced X^2 of best position')
            print(lens_dict, 'lens result')
            print(source_dict, 'source result')
            print(lens_light_dict, 'lens light result')
            print(else_dict, 'else result')
            time_end = time.time()
            print(time_end - time_start, 'time used for PSO', print_key)
        return lens_dict, source_dict, lens_light_dict, else_dict, [X2_list, pos_list, vel_list, []]

    def mcmc_emcee(self, n_walkers, n_run, n_burn, mean_start, sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        sampler = emcee.EnsembleSampler(n_walkers, self.param.num_param(), self.chain.X2_chain_image)
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        new_pos, _, _, _ = sampler.run_mcmc(p0, n_burn)
        sampler.reset()

        store = InMemoryStorageUtil()
        for pos, prob, _, _ in sampler.sample(new_pos, iterations=n_run):
            store.persistSamplingValues(pos, prob, None)

        return store.samples

    def mcmc_CH(self, walkerRatio, n_run, n_burn, mean_start, sigma_start, threadCount=1, init_pos=None, mpi_monch=False):
        """
        runs mcmc on the parameter space given parameter bounds with CosmoHammerSampler
        returns the chain
        """
        lowerLimit, upperLimit = self.param.param_bounds()
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
        if mpi_monch is True:
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
            print('Burn-in itterations: ', n_burn)
            print('Sampling itterations:', n_run)
        sampler.startSampling()
        if sampler.isMaster():
            time_end = time.time()
            print(time_end - time_start, 'time taken for MCMC sampling')
        # if sampler._sampler.pool is not None:
        #     sampler._sampler.pool.close()
        try:
            shutil.rmtree(temp_dir)
        except Exception as ex:
            print(ex)
            pass
        #samples = np.loadtxt(file_prefix+".out")
        #prob = np.loadtxt(file_prefix+"prob.out")
        return store.samples, store.prob
        #return samples, prob


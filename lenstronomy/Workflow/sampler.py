__author__ = 'sibirrer'

# this file contains a class to manage the finder que.
# The job of this class is to select a finder que and to return it in a standardised format

import copy
from lenstronomy.FunctionSet.dipole import Dipole_util

class MCMCQue(object):

    def __init__(self, position_finder, kwargs_finder, samples=None):
        self.position_finder = position_finder
        self.num_cpu = kwargs_finder['num_cpu']
        self.mpi_monch = kwargs_finder.get('mpi_monch', False)
        self.samples = samples
        self.dipole_util = Dipole_util()
        self.dist = 0

    def sample_que(self, kwargs_finder, lens_result, source_result, psf_result, lens_light_result, else_result):
        que_name = kwargs_finder.get('sample_que', 'MCMC')

        if que_name == 'MCMC':
            walkerRatio = kwargs_finder['walkerRatio']
            n_burn = kwargs_finder['n_burn']
            n_run = kwargs_finder['n_run']
            if 'coeffs' in lens_result:
                coeffs = copy.deepcopy(lens_result['coeffs'])
            if kwargs_finder.get('Lens_perturb', False) is True:
                self.samples, self.param_list_mcmc, self.dist = self.position_finder.mcmc_run(lens_result, source_result, lens_light_result, else_result, n_burn, n_run, walkerRatio, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, fix_lens_light=kwargs_finder.get('fix_lens_light', True))
            else:
                self.samples, self.param_list_mcmc, self.dist = self.position_finder.mcmc_arc(lens_result, source_result, lens_light_result, else_result, n_burn, n_run, walkerRatio, numThreads=self.num_cpu, mpi_monch=self.mpi_monch, fix_lens_light=kwargs_finder.get('fix_lens_light', False))
            if 'coeffs' in lens_result:
                lens_result['coeffs'] = coeffs
        if que_name == 'MCMC_lens_light':
            walkerRatio = kwargs_finder['walkerRatio']
            n_burn = kwargs_finder['n_burn']
            n_run = kwargs_finder['n_run']
            self.samples, self.param_list_mcmc, self.dist = self.position_finder.mcmc_lens_light(lens_result, source_result, lens_light_result, else_result, n_burn, n_run, walkerRatio, subgrid_res=2, numThreads=self.num_cpu, mpi_monch=self.mpi_monch)

        return self.samples, self.param_list_mcmc, self.dist


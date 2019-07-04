__author__ = 'aymgal'

import os
import shutil
import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler
import lenstronomy.Util.sampling_util as utils

import dynesty
import dynesty.utils as dyfunc

class DynestySampler(NestedSampler):
    """
    Wrapper for dynamical nested sampling algorithm Dynesty by J. Speagle
    
    paper : https://arxiv.org/abs/1904.02180
    doc : https://dynesty.readthedocs.io/
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None, width_scale=1, sigma_scale=1,
                 bound='multi', sample='auto', use_mpi=False, use_pool={}):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param bound: specific to Dynesty, see https://dynesty.readthedocs.io
        :param sample: specific to Dynesty, see https://dynesty.readthedocs.io
        :param use_mpi: Use MPI computing if `True`
        :param use_pool: specific to Dynesty, see https://dynesty.readthedocs.io
        """
        super(DynestySampler, self).__init__(likelihood_module, prior_type, 
                                             prior_means, prior_sigmas,
                                             width_scale, sigma_scale)

        # create the Dynesty sampler
        if use_mpi:
            from schwimmbad import MPIPool
            import sys

            pool =  MPIPool(use_dill=True)
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            self._sampler = dynesty.DynamicNestedSampler(self.log_likelihood,
                                                         self.prior, self.n_dims,
                                                         bound=bound,
                                                         sample=sample,
                                                         pool=pool,
                                                         use_pool=use_pool)
        else:
            self._sampler = dynesty.DynamicNestedSampler(self.log_likelihood,
                                                         self.prior,
                                                         self.n_dims,
                                                         bound=bound,
                                                         sample=sample)
        self._has_warned = False


    def prior(self, u):
        """
        compute the mapping between the unit cube and parameter cube

        :param u: unit hypercube, sampled by the algorithm
        :return: hypercube in parameter space
        """
        if self.prior_type == 'gaussian':
            p = utils.cube2args_gaussian(u, self.lowers, self.uppers,
                                         self.means, self.sigmas, self.n_dims,
                                         copy=True)
        elif self.prior_type == 'uniform':
            p = utils.cube2args_uniform(u, self.lowers, self.uppers, 
                                        self.n_dims, copy=True)
        return p


    def log_likelihood(self, x):
        """
        compute the log-likelihood given list of parameters

        :param x: parameter values
        :return: log-likelihood (from the likelihood module)
        """
        logL, _ = self._ll(x)
        if not np.isfinite(logL):
            if not self._has_warned:
                print("WARNING : logL is not finite : return very low value instead")
            logL = -1e15
            self._has_warned = True
        return float(logL)


    def run(self, kwargs_run):
        """
        run the Dynesty nested sampler

        see https://dynesty.readthedocs.io for content of kwargs_run

        :param kwargs_run: kwargs directly passed to DynamicNestedSampler.run_nested
        :return: samples, means, logZ, logZ_err, logL, results
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
    
        self._sampler.run_nested(**kwargs_run)

        results = self._sampler.results
        samples_w = results.samples  # weighted samples
        logL = results.logl
        logZ = results.logz
        logZ_err = results.logzerr

        # Compute weighted mean and covariance.
        weights = np.exp(results.logwt - logZ[-1])  # normalized weights
        if np.sum(weights) != 1.:
            # TODO : clearly this is not optimal...
            # weights should by definition be normalized, but it appears that for very small 
            # number of live points (typically in test routines), 
            # it is not *quite* the case (up to 6 decimals)
            weights = weights / np.sum(weights)

        means, covs = dyfunc.mean_and_cov(samples_w, weights)

        # Resample weighted samples to get equally weighted (aka unweighted) samples
        samples = dyfunc.resample_equal(samples_w, weights)

        return samples, means, logZ, logZ_err, logL, results

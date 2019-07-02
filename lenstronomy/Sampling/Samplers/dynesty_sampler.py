__author__ = 'aymgal'

import os
import shutil
import numpy as np

import lenstronomy.Util.sampling_util as utils

import dynesty
import dynesty.utils as dyfunc

class DynestySampler(object):
    """
    Wrapper for dynamical nested sampling algorithm Dynesty by J. Speagle
    
    paper : https://arxiv.org/abs/1904.02180
    doc : https://dynesty.readthedocs.io/
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None,
                 bound='multi', sample='auto', use_mpi=False, use_pool={}):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param bound: specific to Dynesty, see https://dynesty.readthedocs.io
        :param sample: specific to Dynesty, see https://dynesty.readthedocs.io
        """
        self._ll = likelihood_module
        self.lowers, self.uppers = self._ll.param_limits
        self.n_dims, self.param_names = self._ll.param.num_param()

        if prior_type == 'gaussian':
            if prior_means is None or prior_sigmas is None:
                raise ValueError("For gaussian prior type, means and sigmas are required")
            self.means, self.sigmas = prior_means, prior_sigmas
        elif prior_type != 'uniform':
            raise ValueError("Sampling type {} not supported".format(prior_type))
        self.prior_type = prior_type

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
        :return: samples, means, logZ, logZ_err, logL
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
    
        self._sampler.run_nested(**kwargs_run)

        results = self._sampler.results
        samples = results.samples  # TODO : check if it's 'equal weight' or not
        logL = results.logl
        logZ = results.logz
        logZ_err = results.logzerr

        # Compute 5%-95% quantiles.
        # quantiles = dyfunc.quantile(samples, [0.05, 0.95], weights=weights)

        # Compute weighted mean and covariance.
        weights = np.exp(results.logwt - logZ[-1])  # normalized weights
        means, covs = dyfunc.mean_and_cov(samples, weights)

        # Resample weighted samples.
        # samples_equal = dyfunc.resample_equal(samples, weights)

        # Generate a new set of results with statistical+sampling uncertainties.
        # results_sim = dyfunc.simulate_run(results)

        return samples, means, logZ, logZ_err, logL, results

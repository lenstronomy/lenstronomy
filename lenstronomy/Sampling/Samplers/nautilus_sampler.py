__author__ = 'aymgal, johannesulf'

import numpy as np
import lenstronomy.Util.sampling_util as utils
import time

from inspect import signature
from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler


__all__ = ['NautilusSampler']


class NautilusSampler(NestedSampler):
    """
    Wrapper for the nautilus sampler by Johannes U. Lange.

    paper : https://arxiv.org/abs/2306.16923
    doc : https://nautilus-sampler.readthedocs.io
    """

    def __init__(self, likelihood_module, prior_type='uniform',
                 prior_means=None, prior_sigmas=None, width_scale=1,
                 sigma_scale=1, mpi=False, **kwargs):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param mpi: Use MPI computing if `True`
        :param kwargs: kwargs directly passed to Sampler
        """
        self._check_install()
        super(NautilusSampler, self).__init__(likelihood_module, prior_type,
                                              prior_means, prior_sigmas,
                                              width_scale, sigma_scale)

        if mpi:
            from schwimmbad import MPIPool
            import sys

            # use_dill=True not supported for some versions of schwimmbad
            pool = MPIPool(use_dill=True)
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            kwargs['pool'] = pool

        keys = [p.name for p in signature(
            self._nautilus.Sampler).parameters.values()]
        kwargs = {key: kwargs[key] for key in kwargs.keys() & keys}

        self._sampler = self._nautilus.Sampler(
            self.prior, self.log_likelihood, self.n_dims, **kwargs)
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
        else:
            raise ValueError(
                'prior type %s not supported! Chose "gaussian" or "uniform".')
        return p

    def log_likelihood(self, x):
        """
        compute the log-likelihood given list of parameters

        :param x: parameter values
        :return: log-likelihood (from the likelihood module)
        """
        log_l = self._ll(x)
        if not np.isfinite(log_l):
            if not self._has_warned:
                print("WARNING : logL is not finite : return very low value instead")
            log_l = -1e15
            self._has_warned = True
        return float(log_l)

    def run(self, **kwargs):
        """
        run the nautilus nested sampler

        see https://nautilus-sampler.readthedocs.io for content of kwargs

        :param kwargs: kwargs directly passed to Sampler.run
        :return: samples, means, logZ, logZ_err, logL, results
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)

        keys = [p.name for p in signature(
            self._sampler.run).parameters.values()]
        kwargs = {key: kwargs[key] for key in kwargs.keys() & keys}

        self._sampler.run(**kwargs)

        time_start = time.time()
        self._sampler.run(**kwargs)
        points, log_w, log_l = self._sampler.posterior()
        log_z = self._sampler.evidence()
        time_end = time.time()
        print(time_end - time_start, 'time taken for MCMC sampling')
        return points, log_w, log_l, log_z

    def _check_install(self):
        try:
            import nautilus
        except ImportError:
            print("Warning : nautilus not properly installed. \
                  You can get it with $pip install nautilus-sampler.")
        else:
            self._nautilus = nautilus

__author__ = "aymgal, johannesulf"


from inspect import signature
import numpy as np
from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler

__all__ = ["NautilusSampler"]


class NautilusSampler(NestedSampler):
    """Wrapper for the nautilus sampler by Johannes U. Lange.

    paper : https://arxiv.org/abs/2306.16923
    doc : https://nautilus-sampler.readthedocs.io
    """

    def __init__(
        self,
        likelihood_module,
        prior_type="uniform",
        prior_means=None,
        prior_sigmas=None,
        width_scale=1,
        sigma_scale=1,
        mpi=False,
        **kwargs
    ):
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
        super(NautilusSampler, self).__init__(
            likelihood_module,
            prior_type,
            prior_means,
            prior_sigmas,
            width_scale,
            sigma_scale,
        )

        if mpi:
            from schwimmbad import MPIPool
            import sys

            # use_dill=True not supported for some versions of schwimmbad
            pool = MPIPool(use_dill=True)
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            kwargs["pool"] = pool

        keys = [p.name for p in signature(self._nautilus.Sampler).parameters.values()]
        kwargs = {key: kwargs[key] for key in kwargs.keys() & keys}

        self._sampler = self._nautilus.Sampler(
            self.prior, self.log_likelihood, self.n_dims, **kwargs
        )

    def run(self, **kwargs):
        """Run the nautilus nested sampler.

        see https://nautilus-sampler.readthedocs.io for content of kwargs

        :param kwargs: kwargs directly passed to Sampler.run
        :return: samples, means, log_z, log_z_err, log_l, results
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)

        keys = [p.name for p in signature(self._sampler.run).parameters.values()]
        kwargs = {key: kwargs[key] for key in kwargs.keys() & keys}
        self._sampler.run(**kwargs)
        points, log_w, log_l = self._sampler.posterior()
        log_l_raw = np.asarray(log_l)
        log_z = self._sampler.log_z
        log_zerr = log_z / np.sqrt(self._sampler.neff)
        weights = self._normalized_weights(log_w)
        means = np.average(points, weights=weights, axis=0)
        samples, sample_indices = self._resample_equal(points, weights)
        log_l = log_l_raw[sample_indices]
        results = {
            "points": points,
            "log_w": log_w,
            "log_l": log_l_raw,
            "log_z": log_z,
        }

        return samples, means, log_z, log_zerr, log_l, results

    @staticmethod
    def _normalized_weights(log_w):
        """Normalize log weights.

        :param log_w: array of shape (n_samples,)
        :return: normalized weights, array of shape (n_samples,)
        """
        log_w = np.asarray(log_w)
        weights = np.exp(log_w - np.max(log_w))
        return weights / np.sum(weights)

    @staticmethod
    def _resample_equal(points, weights):
        """Resample points with equal weights according to the provided weights.

        :param points: array of shape (n_samples, n_dims)
        :param weights: array of shape (n_samples,)
        :return: resampled points, indices of the resampled points in the original array
        """
        n_samples = len(points)
        positions = (np.arange(n_samples) + 0.5) / n_samples

        # Normalize weights first
        weights_normalized = weights / np.sum(weights)
        cumulative_sum = np.cumsum(weights_normalized)
        cumulative_sum[-1] = 1.0  # For numerical precision

        sample_indices = np.searchsorted(cumulative_sum, positions)
        return points[sample_indices], sample_indices

    def _check_install(self):
        try:
            import nautilus
        except ImportError:
            print("Warning : nautilus not properly installed. \
                  You can get it with $pip install nautilus-sampler.")
        else:
            self._nautilus = nautilus

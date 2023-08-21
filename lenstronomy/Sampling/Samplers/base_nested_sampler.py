__author__ = "aymgal, johannesulf"

import lenstronomy.Util.sampling_util as utils
import numpy as np

__all__ = ["NestedSampler"]


class NestedSampler(object):
    """Base class for nested samplers."""

    def __init__(
        self,
        likelihood_module,
        prior_type,
        prior_means,
        prior_sigmas,
        width_scale,
        sigma_scale,
    ):
        """:param likelihood_module: likelihood_module like in likelihood.py (should be
        callable) :param prior_type: 'uniform' of 'gaussian', for converting the unit
        hypercube to param cube :param prior_means: if prior_type is 'gaussian', mean
        for each param :param prior_sigmas: if prior_type is 'gaussian', std dev for
        each param :param width_scale: scale the widths of the parameters space by this
        factor :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma
        by this factor."""
        self._ll = likelihood_module
        self.n_dims, self.param_names = self._ll.param.num_param()

        lowers, uppers = self._ll.param_limits
        if width_scale < 1:
            self.lowers, self.uppers = utils.scale_limits(lowers, uppers, width_scale)
        else:
            self.lowers, self.uppers = lowers, uppers

        if prior_type == "gaussian":
            if prior_means is None or prior_sigmas is None:
                raise ValueError(
                    "For gaussian prior type, means and sigmas are required"
                )
            self.means, self.sigmas = prior_means, prior_sigmas * sigma_scale
            self.lowers, self.uppers = lowers, uppers
        elif prior_type != "uniform":
            raise ValueError("Sampling type {} not supported".format(prior_type))
        self.prior_type = prior_type
        self._has_warned = False

    def prior(self, u, *args):
        """Compute the mapping between the unit cube and parameter cube.

        :param u: unit hypercube, sampled by the algorithm
        :return: hypercube in parameter space
        """

        # MultiNest passes its own type. Make a copy of the original and create
        # a Python version.
        u_orig = u
        u = np.array([u[i] for i in range(self.n_dims)])

        if self.prior_type == "gaussian":
            p = utils.cube2args_gaussian(
                u,
                self.lowers,
                self.uppers,
                self.means,
                self.sigmas,
                self.n_dims,
                copy=True,
            )
        elif self.prior_type == "uniform":
            p = utils.cube2args_uniform(
                u, self.lowers, self.uppers, self.n_dims, copy=True
            )
        else:
            raise ValueError(
                'prior type %s not supported! Chose "gaussian" or "uniform".'
            )

        # MultiNest expects that we modify the origal array instead of
        # returning the transformed parameters.
        for i in range(self.n_dims):
            # But for PolyChord, this is read-only.
            try:
                u_orig[i] = p[i]
            except ValueError:
                pass

        return p

    def log_likelihood(self, p, *args):
        """Compute the log-likelihood given list of parameters.

        :param x: parameter values
        :return: log-likelihood (from the likelihood module)
        """

        # MultiNest passes its own type.
        p = np.array([p[i] for i in range(self.n_dims)])

        log_l = self._ll(p)
        if not np.isfinite(log_l):
            if not self._has_warned:
                print("WARNING : logL is not finite : return very low value instead")
            log_l = -1e15
            self._has_warned = True
        return float(log_l)

    def run(self, kwargs_run):
        """Run the nested sampling algorithm."""
        raise NotImplementedError("Method not be implemented in base class")

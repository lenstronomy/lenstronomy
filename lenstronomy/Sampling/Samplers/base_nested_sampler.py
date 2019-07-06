__author__ = 'aymgal'

import lenstronomy.Util.sampling_util as utils


class NestedSampler(object):
    """
    Base class for nested samplers
    """

    def __init__(self, likelihood_module, prior_type,
                 prior_means, prior_sigmas, width_scale, sigma_scale):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        """
        self._ll = likelihood_module
        self.n_dims, self.param_names = self._ll.param.num_param()

        lowers, uppers = self._ll.param_limits
        if width_scale < 1:
            self.lowers, self.uppers = utils.scale_limits(lowers, uppers, width_scale)
        else:
            self.lowers, self.uppers = lowers, uppers

        if prior_type == 'gaussian':
            if prior_means is None or prior_sigmas is None:
                raise ValueError("For gaussian prior type, means and sigmas are required")
            self.means, self.sigmas  = prior_means, prior_sigmas * sigma_scale
            self.lowers, self.uppers = lowers, uppers
        elif prior_type != 'uniform':
            raise ValueError("Sampling type {} not supported".format(prior_type))
        self.prior_type = prior_type
        self._has_warned = False


    def prior(self, u):
        """
        compute the mapping between the unit cube and parameter cube

        :param u: unit hypercube, sampled by the algorithm
        :return: hypercube in parameter space
        """
        raise NotImplementedError("Method not be implemented in base class")


    def log_likelihood(self, x):
        """
        compute the log-likelihood given list of parameters

        :param x: parameter values
        :return: log-likelihood (from the likelihood module)
        """
        raise NotImplementedError("Method not be implemented in base class")


    def run(self, kwargs_run):
        """run the nested sampling algorithm"""
        raise NotImplementedError("Method not be implemented in base class")

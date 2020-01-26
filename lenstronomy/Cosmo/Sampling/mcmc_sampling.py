import emcee
from lenstronomy.Cosmo.Sampling.cosmo_likelihood import CosmoLikelihood
from lenstronomy.Cosmo.Sampling.param_manager import CosmoParam


class MCMCSampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=False,
                 lambda_mst_sampling=False, custom_prior=None):
        """
        initialise the classes of the chain and for parameter options

        :param custom_prior: None or a definition that takes the keywords from the CosmoParam conventions and returns a
        log likelihood value (e.g. prior)
        """
        self.chain = CosmoLikelihood(kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed,
                                     ppn_sampling=ppn_sampling, lambda_mst_sampling=lambda_mst_sampling,
                                     custom_prior=custom_prior)
        self.cosmoParam = CosmoParam(cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=ppn_sampling,
                                     lambda_mst_sampling=lambda_mst_sampling)

    def mcmc_emcee(self, n_walkers, n_burn, n_run, kwargs_mean_start, kwargs_sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        num_param = self.cosmoParam.num_param
        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.likelihood, args=())
        mean_start = self.cosmoParam.kwargs2args(kwargs_mean_start)
        sigma_start = self.cosmoParam.kwargs2args(kwargs_sigma_start)
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        sampler.run_mcmc(p0, n_burn+n_run, progress=True)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        return flat_samples

    def param_names(self, latex_style=False):
        """
        list of parameter names being sampled in the same order as teh sampling

        :param latex_style: bool, if True returns strings in latex symbols, else in the convention of the sampler
        :return: list of strings
        """
        labels = self.cosmoParam.param_list(latex_style=latex_style)
        return labels

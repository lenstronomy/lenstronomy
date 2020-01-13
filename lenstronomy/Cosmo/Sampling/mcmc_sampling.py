import emcee
from lenstronomy.Cosmo.Sampling.cosmo_likelihood import CosmoLikelihood
from lenstronomy.Cosmo.Sampling.param_manager import CosmoParam


class MCMCSampler(object):
    """
    class which executes the different sampling  methods
    """
    def __init__(self, kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=False):
        """
        initialise the classes of the chain and for parameter options
        """
        self.chain = CosmoLikelihood(kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling)
        self.cosmoParam = CosmoParam(cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=ppn_sampling)

    def mcmc_emcee(self, n_walkers, n_burn, n_run, kwargs_mean_start, kwargs_sigma_start):
        """
        returns the mcmc analysis of the parameter space
        """
        num_param, param_names = self.cosmoParam.num_param
        sampler = emcee.EnsembleSampler(n_walkers, num_param, self.chain.likelihood, args=())
        mean_start = self.cosmoParam.kwargs2args(kwargs_mean_start)
        sigma_start = self.cosmoParam.kwargs2args(kwargs_sigma_start)
        p0 = emcee.utils.sample_ball(mean_start, sigma_start, n_walkers)
        sampler.run_mcmc(p0, n_burn+n_run, progress=True)
        flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
        return flat_samples

from lenstronomy.Cosmo.Sampling.lens_likelihood import LensSampleLikelihood
from lenstronomy.Cosmo.Sampling.param_manager import CosmoParam
import numpy as np


class CosmoLikelihood(object):
    """
    this class contains the likelihood function of the Strong lensing analysis
    """

    def __init__(self, kwargs_lens_list, cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=False,
                 lambda_mst_sampling=False, custom_prior=None):
        """

        :param kwargs_lens_list: keyword argument list specifying the arguments of the LensLikelihood class
        :param cosmology: string describing cosmological model
        :param ppn_sampling:post-newtonian parameter sampling
        :param custom_prior: None or a definition that takes the keywords from the CosmoParam conventions and returns a
        log likelihood value (e.g. prior)
        """
        self._cosmology = cosmology
        self._kwargs_lens_list = kwargs_lens_list
        self._likelihoodLensSample = LensSampleLikelihood(kwargs_lens_list)
        self._param = CosmoParam(cosmology, kwargs_lower, kwargs_upper, kwargs_fixed, ppn_sampling=ppn_sampling,
                                 lambda_mst_sampling=lambda_mst_sampling)
        self._lowerlimit, self._upperlimit = self._param.param_bounds
        self._prior_add = False
        if custom_prior is not None:
            self._prior_add = True
        self._custom_prior = custom_prior

    def likelihood(self, args):
        """

        :param args: list of sampled parameters
        :return: log likelihood of the combined lenses
        """
        for i in range(0, len(args)):
            if args[i] < self._lowerlimit[i] or args[i] > self._upperlimit[i]:
                return -np.inf

        kwargs = self._param.args2kwargs(args)
        if self._cosmology == "oLCDM":
            # assert we are not in a crazy cosmological situation that prevents computing the angular distance integral
            h0, ok, om = kwargs['h0'], kwargs['ok'], kwargs['om']
            if np.any(
                    [ok * (1.0 + lens['z_source']) ** 2 + om * (1.0 + lens['z_source']) ** 3 + (1.0 - om - ok) <= 0 for lens in
                     self._kwargs_lens_list]):
                return -np.inf
            # make sure that Omega_DE is not negative...
            if 1.0 - om - ok <= 0:
                return -np.inf
        cosmo = self._param.cosmo(kwargs)
        logL = self._likelihoodLensSample.log_likelihood(cosmo=cosmo, gamma_ppn=kwargs.get('gamma_ppn', 1),
                                                         kappa_ext=kwargs.get('kappa_ext', 0),
                                                         lambda_mst=kwargs.get('lambda_mst', 1))
        if self._prior_add is True:
            logL += self._custom_prior(kwargs)
        return logL

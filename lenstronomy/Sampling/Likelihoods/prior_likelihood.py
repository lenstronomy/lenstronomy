import numpy as np


class PriorLikelihood(object):
    """
    class containing additional Gaussian priors to be folded into the likelihood

    """
    def __init__(self, prior_lens=[], prior_source=[], prior_lens_light=[], prior_ps=[], prior_cosmo=[]):
        """

        :param prior_lens: list of [index_model, param_name, value, 1-sigma priors]
        :param prior_source: list of [index_model, param_name, value, 1-sigma priors]
        :param prior_lens_light: list of [index_model, param_name, value, 1-sigma priors]
        :param prior_ps: list of [index_model, param_name, value, 1-sigma priors]
        :param prior_cosmo: list of [index_model, param_name, value, 1-sigma priors]
        """
        self._prior_lens, self._prior_source, self._prior_lens_light, self._prior_ps, self._prior_cosmo = \
            prior_lens, prior_source, prior_lens_light, prior_ps, prior_cosmo

    def logL(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps, kwargs_cosmo):
        """

        :param kwargs_lens: lens model parameter list
        :return: log likelihood of lens center
        """
        logL = 0
        logL += self._prior_kwargs_list(kwargs_lens, self._prior_lens)
        logL += self._prior_kwargs_list(kwargs_source, self._prior_source)
        logL += self._prior_kwargs_list(kwargs_lens_light, self._prior_lens_light)
        logL += self._prior_kwargs_list(kwargs_ps, self._prior_ps)
        logL += self._prior_kwargs(kwargs_cosmo, self._prior_cosmo)
        return logL

    def _prior_kwargs_list(self, kwargs_list, prior_list):
        """

        :param kwargs_list: keyword argument list
        :param prior_list: prior list
        :return: logL
        """
        logL = 0
        for i in range(len(prior_list)):
            index, param_name, value, sigma = prior_list[i]
            model_value = kwargs_list[index][param_name]
            dist = (model_value - value) ** 2 / sigma ** 2 / 2
            logL -= np.sum(dist)
        return logL

    def _prior_kwargs(self, kwargs, prior_list):
        """
        prior computation for a keyword argument (not list thereof)

        :param kwargs: keyword argument
        :return: logL
        """
        logL = 0
        for i in range(len(prior_list)):
            param_name, value, sigma = prior_list[i]
            model_value = kwargs[param_name]
            dist = (model_value - value) ** 2 / sigma ** 2 / 2
            logL -= np.sum(dist)
        return logL

import numpy as np
from lenstronomy.Util.prob_density import KDE1D

__all__ = ['PriorLikelihood']


class PriorLikelihood(object):
    """
    class containing additional Gaussian priors to be folded into the likelihood

    """
    def __init__(self, prior_lens=[], prior_source=[], prior_lens_light=[], prior_ps=[], prior_special=[],
                 prior_extinction=[], prior_lens_kde=[], prior_source_kde=[], prior_lens_light_kde=[], prior_ps_kde=[],
                 prior_special_kde=[], prior_extinction_kde=[],
                 prior_lens_lognormal=[], prior_source_lognormal=[],
                 prior_lens_light_lognormal=[],
                 prior_ps_lognormal=[], prior_special_lognormal=[],
                 prior_extinction_lognormal=[],
                 ):
        """

        :param prior_lens: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_source: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_lens_light: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_ps: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_special: list of [param_name, mean, 1-sigma priors]
        :param prior_extinction: list of [index_model, param_name, mean, 1-sigma priors]

        :param prior_lens_kde: list of [index_model, param_name, samples]
        :param prior_source_kde: list of [index_model, param_name, samples]
        :param prior_lens_light_kde: list of [index_model, param_name, samples]
        :param prior_ps_kde: list of [index_model, param_name, samples]
        :param prior_special_kde: list of [param_name, samples]
        :param prior_extinction_kde: list of [index_model, param_name, samples]

        :param prior_lens_lognormal: list of [index_model, param_name, mean, 1-sigma
        priors]
        :param prior_source_lognormal: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_lens_light_lognormal: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_ps_lognormal: list of [index_model, param_name, mean, 1-sigma priors]
        :param prior_special_lognormal: list of [param_name, mean, 1-sigma priors]
        :param prior_extinction_lognormal: list of [index_model, param_name, mean, 1-sigma priors]

        """

        self._prior_lens, self._prior_source, self._prior_lens_light, self._prior_ps, self._prior_special, self._prior_extinction = \
            prior_lens, prior_source, prior_lens_light, prior_ps, prior_special, prior_extinction
        self._prior_lens_kde, self._prior_source_kde, self._prior_lens_light_kde, self._prior_ps_kde = prior_lens_kde, \
                                                                                                       prior_source_kde, \
                                                                                                       prior_lens_light_kde, \
                                                                                                       prior_ps_kde
        self._prior_lens_lognormal, self._prior_source_lognormal, \
        self._prior_lens_light_lognormal, \
        self._prior_ps_lognormal, self._prior_special_lognormal, \
        self._prior_extinction_lognormal = \
            prior_lens_lognormal, prior_source_lognormal, \
            prior_lens_light_lognormal, prior_ps_lognormal, \
            prior_special_lognormal, prior_extinction_lognormal

        self._kde_lens_list = self._init_kde(prior_lens_kde)
        self._kde_source_list = self._init_kde(prior_source_kde)
        self._kde_lens_light_list = self._init_kde(prior_lens_light_kde)
        self._kde_ps_list = self._init_kde(prior_ps_kde)
        self._kde_lens_light_list = self._init_kde(prior_lens_light_kde)

    def _init_kde(self, prior_list_kde):
        """

        :param prior_list_kde: list of [index_model, param_name, samples]
        :return: list of initiated KDE's
        """
        kde_list = []
        for prior in prior_list_kde:
            index, param_name, samples = prior
            kde_list.append(KDE1D(values=samples))
        return kde_list

    def logL(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
             kwargs_extinction=None):
        """

        :param kwargs_lens: lens model parameter list
        :return: log likelihood of lens center
        """
        logL = 0
        logL += self._prior_kwargs_list(kwargs_lens, self._prior_lens)
        logL += self._prior_kwargs_list(kwargs_source, self._prior_source)
        logL += self._prior_kwargs_list(kwargs_lens_light, self._prior_lens_light)
        logL += self._prior_kwargs_list(kwargs_ps, self._prior_ps)
        logL += self._prior_kwargs(kwargs_special, self._prior_special)
        logL += self._prior_kwargs_list(kwargs_extinction, self._prior_extinction)

        logL += self._prior_lognormal_kwargs_list(kwargs_lens,
                                                  self._prior_lens_lognormal)
        logL += self._prior_lognormal_kwargs_list(kwargs_source, self._prior_source_lognormal)
        logL += self._prior_lognormal_kwargs_list(kwargs_lens_light,
                                        self._prior_lens_light_lognormal)
        logL += self._prior_lognormal_kwargs_list(kwargs_ps, self._prior_ps_lognormal)
        logL += self._prior_lognormal_kwargs(kwargs_special, self._prior_special_lognormal)
        logL += self._prior_lognormal_kwargs_list(kwargs_extinction,
                                        self._prior_extinction_lognormal)

        logL += self._prior_kde_list(kwargs_lens, self._prior_lens_kde, self._kde_lens_list)
        logL += self._prior_kde_list(kwargs_source, self._prior_source_kde, self._kde_source_list)
        logL += self._prior_kde_list(kwargs_lens_light, self._prior_lens_light_kde, self._kde_lens_light_list)
        logL += self._prior_kde_list(kwargs_ps, self._prior_ps_kde, self._kde_ps_list)
        return logL

    def _prior_kde_list(self, kwargs_list, prior_list, kde_list):
        """

        :param kwargs_list:
        :param prior_list:
        :return:
        """
        logL = 0
        for i in range(len(prior_list)):
            index, param_name, values = prior_list[i]
            model_value = kwargs_list[index][param_name]
            likelihood = kde_list[i].likelihood(model_value)[0]
            logL += np.log(likelihood)
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

    def _prior_lognormal_kwargs_list(self, kwargs_list, prior_list):
        """

        :param kwargs_list: keyword argument list
        :param prior_list: prior list
        :return: logL
        """
        logL = 0
        for i in range(len(prior_list)):
            index, param_name, value, sigma = prior_list[i]
            model_value = kwargs_list[index][param_name]
            dist = (np.log(model_value) - value) ** 2 / sigma ** 2 / 2 + model_value
            logL -= np.sum(dist)
        return logL

    def _prior_lognormal_kwargs(self, kwargs, prior_list):
        """
        prior computation for a keyword argument (not list thereof)

        :param kwargs: keyword argument
        :return: logL
        """
        logL = 0
        for i in range(len(prior_list)):
            param_name, value, sigma = prior_list[i]
            model_value = kwargs[param_name]
            dist = (np.log(model_value) - value) ** 2 / sigma ** 2 / 2 + model_value
            logL -= np.sum(dist)
        return logL

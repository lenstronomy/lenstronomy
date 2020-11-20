import numpy as np
from lenstronomy.Util import class_creator

__all__ = ['ImageLikelihood']


class ImageLikelihood(object):
    """
    manages imaging data likelihoods
    """

    def __init__(self, multi_band_list, multi_band_type, kwargs_model, bands_compute=None, image_likelihood_mask_list=None,
                 source_marg=False, linear_prior=None, check_positive_flux=False, kwargs_pixelbased=None):
        """

        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the fitting
        :param image_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be evaluated in the likelihood
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the inferred
        linear coefficients
        :param linear_prior: float or list of floats (when multi-linear setting is chosen) indicating the range of
        linear amplitude priors when computing the marginalization term.
        :param force_minimum_source_surface_brightness: bool, if True, evaluates the source surface brightness on a grid
        and evaluates if all positions exceed the minimum flux
        :param flux_min: float, minimum flux (surface brightness to obey when force_minimum_source_brightness is enabled
        :param check_positive_flux: bool, option to punish models that do not have all positive linear amplitude parameters
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver (see SLITronomy documentation)
        """
        self.imSim = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model,
                                                 bands_compute=bands_compute, likelihood_mask_list=image_likelihood_mask_list,
                                                 kwargs_pixelbased=kwargs_pixelbased)
        self._model_type = self.imSim.type
        self._source_marg = source_marg
        self._linear_prior = linear_prior
        self._check_positive_flux = check_positive_flux

    def logL(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
             kwargs_extinction=None):
        """

        :param kwargs_lens: lens model keyword argument list according to LensModel module
        :param kwargs_source: source light keyword argument list according to LightModel module
        :param kwargs_lens_light: deflector light (not lensed) keyword argument list according to LightModel module
        :param kwargs_ps: point source keyword argument list according to PointSource module
        :param kwargs_special: special keyword argument list as part of the Param module
        :param kwargs_extinction: extinction parameter keyword argument list according to LightModel module
        :return: log likelihood of the data given the model
        """
        logL = self.imSim.likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                      kwargs_extinction=kwargs_extinction, kwargs_special=kwargs_special,
                                                      source_marg=self._source_marg, linear_prior=self._linear_prior,
                                                      check_positive_flux=self._check_positive_flux)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL

    @property
    def num_data(self):
        """

        :return: number of image data points
        """
        return self.imSim.num_data_evaluate

    def num_param_linear(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None,
                         kwargs_special=None, kwargs_extinction=None):
        """

        :return:  number of linear parameters solved for during the image reconstruction process
        """
        return self.imSim.num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

    def reset_point_source_cache(self, bool=True):
        """

        :param bool: boolean
        :return:
        """
        self.imSim.reset_point_source_cache(bool=bool)


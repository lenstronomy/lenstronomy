import numpy as np
from lenstronomy.Util import class_creator


class ImageLikelihood(object):
    """
    manages imaging data likelihoods
    """

    def __init__(self, multi_band_list, multi_band_type, kwargs_model, bands_compute=None, likelihood_mask_list=None,
                 source_marg=False, force_minimum_source_surface_brightness=False, flux_min=0):
        """

        :param imSim_class: instance of a class that simulates one (or more) images and returns the likelihood, such as
        ImageModel(), Multiband(), MulitExposure()
        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the fitting
        :param likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be evaluated in the likelihood
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the infered
        linear coefficients
        :param force_minimum_source_surface_brightness: bool, if True, evaluates the source surface brightness on a grid
        and evaluates if all positions exceed the minimum flux
        :param flux_min: float, minimum flux (surface brightness to obey when force_minimum_source_brightness is enabled
        """

        self.imSim = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model, bands_compute=bands_compute,
                                   likelihood_mask_list=likelihood_mask_list, band_index=0)
        self._model_type = self.imSim.type
        self._source_marg = source_marg
        self._force_minimum_source_surface_brightness = force_minimum_source_surface_brightness
        self._flux_min = flux_min

    def logL(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
        """

        :param kwargs_lens:
        :param kwargs_source:
        :param kwargs_lens_light:
        :param kwargs_ps:
        :return:
        """

        logL = self.imSim.likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                      source_marg=self._source_marg)

        if self._force_minimum_source_surface_brightness is True and len(kwargs_source) > 0:
            bool = self._check_minimum_source_flux(kwargs_lens, kwargs_source)
            if bool is True:
                logL -= 10 ** 10
        return logL

    def _check_minimum_source_flux(self, kwargs_lens, kwargs_source):
        if self._model_type in ['single-band']:
            flux = self.imSim.source_surface_brightness(kwargs_source, kwargs_lens=kwargs_lens, unconvolved=True)
            if np.min(flux) < self._flux_min:
                return True
        else:
            raise ValueError("check_mimimum source flux not supported for modelling type %s." % self._model_type)
        return False

    @property
    def num_data(self):
        """

        :return: number of image data points
        """
        return self.imSim.num_data_evaluate

    def num_param_linear(self, kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps):
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


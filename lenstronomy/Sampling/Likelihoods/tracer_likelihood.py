import numpy as np
from lenstronomy.Util import class_creator
from lenstronomy.ImSim.tracer_model import TracerModel


class TracerLikelihood(object):
    """
    class to evaluate the tracer map
    """

    def __init__(self, tracer_band_list, kwargs_model, tracer_bands_compute=None,
                 tracer_likelihood_mask_list=None, tracer_light_model_band=0):
        """

        :param tracer_bands_compute: list of bools with same length as data objects, indicates which "band" to include in the
         fitting
        :param tracer_likelihood_mask_list: list of boolean 2d arrays of size of images marking the pixels to be
         evaluated in the likelihood
        """
        self.image_model = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model,
                                                       bands_compute=bands_compute,
                                                       image_likelihood_mask_list=image_likelihood_mask_list,
                                                       kwargs_pixelbased=kwargs_pixelbased, linear_solver=linear_solver)

        self.tracerModel = class_creator.create_tracer_model(tracer_band_list, kwargs_model, tracer_likelihood_mask_list,
                                                             tracer_bands_compute)
        self.image_model = class_creator.create_im_sim(multi_band_list, multi_band_type, kwargs_model,
                                                       bands_compute=bands_compute,
                                                       image_likelihood_mask_list=image_likelihood_mask_list,
                                                       kwargs_pixelbased=kwargs_pixelbased, linear_solver=linear_solver)
        self._tracer_light_model_band = tracer_light_model_band

    def logL(self, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None, kwargs_ps=None, kwargs_special=None,
             kwargs_extinction=None, param=None):
        """

        :param kwargs_lens: lens model keyword argument list according to LensModel module
        :param kwargs_source: source light keyword argument list according to LightModel module
        :param kwargs_lens_light: deflector light (not lensed) keyword argument list according to LightModel module
        :param kwargs_ps: point source keyword argument list according to PointSource module
        :param kwargs_special: special keyword argument list as part of the Param module
        :param kwargs_extinction: extinction parameter keyword argument list according to LightModel module
        :return: log likelihood of the data given the model
        """
        # TODO: get kwargs_source from a specified setting/band, make sure linear inversion has been applied

        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = self.image_model.update_linear_kwargs(param, model_band=self._tracer_light_model_band, kwargs_lens=kwargs_lens,
                                                                                                         kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light,
                                                                                                         kwargs_ps=kwargs_ps)

        logL = self.tracerModel.likelihood_data_given_model(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps,
                                                            kwargs_extinction=kwargs_extinction,
                                                            kwargs_special=kwargs_special)
        if np.isnan(logL) is True:
            return -10 ** 15
        return logL

    @property
    def num_data(self):
        """

        :return: number of image data points
        """
        return self.tracerModel.num_data_evaluate

    def reset_point_source_cache(self, cache=True):
        """

        :param cache: boolean
        :return: None
        """
        self.tracerModel.reset_point_source_cache(cache=cache)
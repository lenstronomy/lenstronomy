from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.Util import util
import numpy as np


class TracerModelSource(ImageModel):
    """
    Tracer model class, inherits ImageModel.

    """
    def __init__(self, data_class, psf_class=None, lens_model_class=None, source_model_class=None,
                 lens_light_model_class=None, point_source_class=None, extinction_class=None,
                 tracer_source_class=None, kwargs_numerics=None, likelihood_mask=None,
                 psf_error_map_bool_list=None, kwargs_pixelbased=None, tracer_partition=None):
        """

        :param data_class: ImageData() instance
        :param psf_class: PSF() instance
        :param lens_model_class: LensModel() instance
        :param source_model_class: LightModel() instance
        :param lens_light_model_class: LightModel() instance
        :param point_source_class: PointSource() instance
        :param tracer_source_class: LightModel() instance describing the tracers of the source
        :param kwargs_numerics: keyword arguments passed to the Numerics module
        :param likelihood_mask: 2d boolean array of pixels to be counted in the likelihood calculation/linear
         optimization
        :param psf_error_map_bool_list: list of boolean of length of point source models.
         Indicates whether PSF error map is used for the point source model stated as the index.
        :param kwargs_pixelbased: keyword arguments with various settings related to the pixel-based solver
         (see SLITronomy documentation) being applied to the point sources.
        :param tracer_partition: in case of tracer models for specific sub-parts of the surface brightness model
         [[list of light profiles, list of tracer profiles], [list of light profiles, list of tracer profiles], [...], ...]
        :type tracer_partition: None or list
        """
        if likelihood_mask is None:
            likelihood_mask = np.ones_like(data_class.data)
        self.likelihood_mask = np.array(likelihood_mask, dtype=bool)
        self._mask1d = util.image2array(self.likelihood_mask)
        if tracer_partition is None:
            tracer_partition = [[None, None]]
        self._tracer_partition = tracer_partition
        super(TracerModelSource, self).__init__(data_class, psf_class=psf_class, lens_model_class=lens_model_class,
                                                source_model_class=source_model_class,
                                                lens_light_model_class=lens_light_model_class,
                                                point_source_class=point_source_class, extinction_class=extinction_class,
                                                kwargs_numerics=kwargs_numerics, kwargs_pixelbased=kwargs_pixelbased)
        if psf_error_map_bool_list is None:
            psf_error_map_bool_list = [True] * len(self.PointSource.point_source_type_list)
        self._psf_error_map_bool_list = psf_error_map_bool_list
        if tracer_source_class is None:
            tracer_source_class = LightModel(light_model_list=[])
        if lens_model_class is None:
            lens_model_class = LensModel(lens_model_list=[])
        self.tracer_mapping = Image2SourceMapping(lensModel=lens_model_class, sourceModel=tracer_source_class)
        self.tracer_source_class = tracer_source_class

    def tracer_model(self, kwargs_tracer_source, kwargs_lens, kwargs_source, kwargs_extinction=None, kwargs_special=None,
                     de_lensed=False):
        """
        tracer model as a convolved surface brightness weighted quantity
        conv(tracer * surface brightness) / conv(surface brightness)

        :param kwargs_tracer_source:
        :param kwargs_lens:
        :param kwargs_source:
        :return: model predicted observed tracer component
        """
        tracer_brightness_conv = np.zeros_like(self.Data.data)
        source_light_conv = np.zeros_like(self.Data.data)
        for [k_light, k_tracer] in self._tracer_partition:
            source_light_k = self._source_surface_brightness_analytical_numerics(kwargs_source, kwargs_lens,
                                                                               kwargs_extinction,
                                                                               kwargs_special=kwargs_special,
                                                                               de_lensed=de_lensed, k=k_light)
            source_light_conv_k = self.ImageNumerics.re_size_convolve(source_light_k, unconvolved=False)
            source_light_conv_k[source_light_conv_k < 10 ** (-20)] = 10 ** (-20)
            tracer_k = self._tracer_model_source(kwargs_tracer_source, kwargs_lens, de_lensed=de_lensed, k=k_tracer)
            tracer_brightness_conv_k = self.ImageNumerics.re_size_convolve(tracer_k * source_light_k, unconvolved=False)
            tracer_brightness_conv += tracer_brightness_conv_k
            source_light_conv += source_light_conv_k
        return tracer_brightness_conv / source_light_conv

    def _tracer_model_source(self, kwargs_tracer_source, kwargs_lens, de_lensed=False, k=None):
        """

        :param kwargs_tracer_source:
        :param kwargs_lens:
        :return:
        """
        ra_grid, dec_grid = self.ImageNumerics.coordinates_evaluate
        if de_lensed is True:
            source_light = self.tracer_source_class.surface_brightness(ra_grid, dec_grid, kwargs_tracer_source, k=k)
        else:
            source_light = self.tracer_mapping.image_flux_joint(ra_grid, dec_grid, kwargs_lens, kwargs_tracer_source, k=k)
        return source_light

    def likelihood_data_given_model(self, kwargs_tracer_source, kwargs_lens, kwargs_source, kwargs_extinction=None,
                                    kwargs_special=None):
        model = self.tracer_model(kwargs_tracer_source, kwargs_lens, kwargs_source, kwargs_extinction, kwargs_special)
        log_likelihood = self.Data.log_likelihood(model, self.likelihood_mask, additional_error_map=0)
        return log_likelihood

    @property
    def num_data_evaluate(self):
        """
        number of data points to be used in the linear solver
        :return:
        """
        return int(np.sum(self.likelihood_mask))

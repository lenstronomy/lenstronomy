import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.MultiBand.multiband import MultiBand
from lenstronomy.ImSim.MultiBand.multiband_multimodel import MultiBandMultiModel
from lenstronomy.ImSim.MultiBand.multi_exposures import MultiExposures
from lenstronomy.ImSim.MultiBand.multi_frame import MultiFrame


class ImageLikelihood(object):
    """
    manages imaging data likelihoods
    """

    def __init__(self, multi_band_list, multi_band_type, lens_model_class, source_model_class, lens_light_model_class,
                     point_source_class, bands_compute=None, source_marg=False,
                 force_minimum_source_surface_brightness=False, flux_min=0):
        """

        :param imSim_class: instance of a class that simulates one (or more) images and returns the likelihood, such as
        ImageModel(), Multiband(), MulitExposure()
        :param bands_compute: list of bools with same length as data objects, indicates which "band" to include in the fitting
        :param source_marg: marginalization addition on the imaging likelihood based on the covariance of the infered
        linear coefficients
        :param force_minimum_source_surface_brightness: bool, if True, evaluates the source surface brightness on a grid
        and evaluates if all positions exceed the minimum flux
        :param flux_min: float, minimum flux (surface brightness to obey when force_minimum_source_brightness is enabled
        """

        self.imSim = create_im_sim(multi_band_list, multi_band_type, lens_model_class, source_model_class,
                                   lens_light_model_class, point_source_class, bands_compute=bands_compute)
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
        return self.imSim.num_data_evaluate()

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


def create_im_sim(multi_band_list, image_type, lens_model_class, source_model_class, lens_light_model_class,
                  point_source_class, bands_compute=None):
    """


    :param image_type: string, option when having multiple imaging data sets modelled simultaneously. Options are:

    - 'multi-band': linear amplitudes are inferred on single data set
    - 'multi-exposure': linear amplitudes ae jointly inferred
    - 'multi-frame': multiple frames (as single exposures with disjoint lens model

    :return: MultiBand class instance
    """

    lens_model_list = lens_model_class.lens_model_list
    source_light_model_list = source_model_class.profile_type_list
    lens_light_model_list = lens_light_model_class.profile_type_list
    if image_type == 'multi-band':
        multiband = MultiBand(multi_band_list, lens_model_class, source_model_class, lens_light_model_class,
                              point_source_class, compute_bool=bands_compute)
    elif image_type == 'multi-band-multi-model':
        multiband = MultiBandMultiModel(multi_band_list, lens_model_class, source_model_list=source_light_model_list,
                                        lens_light_model_list=lens_light_model_list,
                                        point_source_class=point_source_class, compute_bool=bands_compute)
    elif image_type == 'multi-exposure':
        multiband = MultiExposures(multi_band_list, lens_model_class, source_model_class, lens_light_model_class,
                                   point_source_class, compute_bool=bands_compute)
    elif image_type == 'multi-frame':
        multiband = MultiFrame(multi_band_list, lens_model_list, source_model_class, lens_light_model_class,
                               point_source_class, compute_bool=bands_compute)
    elif image_type == 'single-band':
        kwargs_data, kwargs_psf, kwargs_numerics = multi_band_list
        data_class = ImageData(**kwargs_data)
        psf_class = PSF(kwargs_psf)
        multiband = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                            point_source_class, kwargs_numerics)
    else:
        raise ValueError("type %s is not supported!" % image_type)
    return multiband
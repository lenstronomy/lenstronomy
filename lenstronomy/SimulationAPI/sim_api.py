from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.SimulationAPI.model_api import ModelAPI
from lenstronomy.ImSim.image_model import ImageModel

import copy
import numpy as np

__all__ = ['SimAPI']


class SimAPI(DataAPI, ModelAPI):
    """
    This class manages the model parameters in regard of the data specified in SingleBand. In particular,
    this API translates models specified in units of astronomical magnitudes into the amplitude parameters used in the
    LightModel module of lenstronomy.
    Optionally, this class can also handle inputs with cosmology dependent lensing quantities and translates them to
    the optical quantities being used in the lenstronomy LensModel module.
    All other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource modules
    """
    def __init__(self, numpix, kwargs_single_band, kwargs_model):
        """
        
        :param numpix: number of pixels per axis
        :param kwargs_single_band: keyword arguments specifying the class instance of DataAPI 
        :param kwargs_model: keyword arguments specifying the class instance of ModelAPI 
        """
        DataAPI.__init__(self, numpix, **kwargs_single_band)
        ModelAPI.__init__(self, **kwargs_model)

    def image_model_class(self, kwargs_numerics=None):
        """

        :param kwargs_numerics: keyword arguments list of Numerics module
        :return: instance of the ImageModel class with all the specified configurations
        """
        return ImageModel(self.data_class, self.psf_class, self.lens_model_class, self.source_model_class,
                          self.lens_light_model_class, self.point_source_model_class, kwargs_numerics=kwargs_numerics)

    def magnitude2amplitude(self, kwargs_lens_light_mag=None, kwargs_source_mag=None, kwargs_ps_mag=None):
        """
        'magnitude' definition are in APPARENT magnitudes as observed on the sky, not intrinsic!

        :param kwargs_lens_light_mag: keyword argument list as for LightModel module except that 'amp' parameters are
         'magnitude' parameters.
        :param kwargs_source_mag: keyword argument list as for LightModel module except that 'amp' parameters are
         'magnitude' parameters.
        :param kwargs_ps_mag: keyword argument list as for PointSource module except that 'amp' parameters are
         'magnitude' parameters.
        :return: value of the lenstronomy 'amp' parameter such that the total flux of the profile type results in this
         magnitude for all the light models. These keyword arguments conform with the lenstronomy LightModel syntax.
        """

        kwargs_lens_light = copy.deepcopy(kwargs_lens_light_mag)
        if kwargs_lens_light_mag is not None:
            for i, kwargs_mag in enumerate(kwargs_lens_light_mag):
                kwargs_new = kwargs_lens_light[i]
                del kwargs_new['magnitude']
                cps_norm = self.lens_light_model_class.total_flux(kwargs_list=kwargs_lens_light, norm=True, k=i)[0]
                magnitude = kwargs_mag['magnitude']
                cps = self.magnitude2cps(magnitude)
                amp = cps / cps_norm
                kwargs_new['amp'] = amp

        kwargs_source = copy.deepcopy(kwargs_source_mag)
        if kwargs_source_mag is not None:
            for i, kwargs_mag in enumerate(kwargs_source_mag):
                kwargs_new = kwargs_source[i]
                del kwargs_new['magnitude']
                cps_norm = self.source_model_class.total_flux(kwargs_list=kwargs_source, norm=True, k=i)[0]
                magnitude = kwargs_mag['magnitude']
                cps = self.magnitude2cps(magnitude)
                amp = cps / cps_norm
                kwargs_new['amp'] = amp

        kwargs_ps = copy.deepcopy(kwargs_ps_mag)
        if kwargs_ps_mag is not None:
            amp_list = []
            for i, kwargs_mag in enumerate(kwargs_ps_mag):
                kwargs_new = kwargs_ps[i]
                del kwargs_new['magnitude']
                cps_norm = 1
                magnitude = np.array(kwargs_mag['magnitude'])
                cps = self.magnitude2cps(magnitude)
                amp = cps / cps_norm
                amp_list.append(amp)
            kwargs_ps = self.point_source_model_class.set_amplitudes(amp_list, kwargs_ps)
        return kwargs_lens_light, kwargs_source, kwargs_ps

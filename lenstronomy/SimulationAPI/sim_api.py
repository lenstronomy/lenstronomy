from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.SimulationAPI.model_api import ModelAPI
from lenstronomy.ImSim.image_model import ImageModel


class SimAPI(DataAPI, ModelAPI):
    """
    This class manages the model parameters in regard of the data specified in SingleBand. In particular,
    this API translates models specified in units of astronomical magnitudes into the amplitude parameters used in the
    LightModel module of lenstronomy.
    Optionally, this class can also handle inputs with cosmology dependent lensing quantities and translates them to
    the optical quantities being used in the lenstronomy LensModel module.
    All other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource modules
    """
    def __init__(self, numpix, kwargs_single_band, kwargs_model, kwargs_numerics):
        """
        
        :param numpix: number of pixels per axis
        :param kwargs_single_band: keyword arguments specifying the class instance of DataAPI 
        :param kwargs_model: keyword arguments specifying the class instance of ModelAPI 
        :param kwargs_numerics: keyword argument with various numeric description (see ImageNumerics class for options)
        """
        DataAPI.__init__(self, numpix, **kwargs_single_band)
        ModelAPI.__init__(self, **kwargs_model)
        self._image_model_class = ImageModel(self.data_class, self.psf_class, self.lens_model_class,
                                             self.source_model_class, self.lens_light_model_class,
                                             self.point_source_model_class, kwargs_numerics)

    @property
    def image_model_class(self):
        """

        :return: instance of the ImageModel class with all the specified configurations
        """
        return self._image_model_class

    def magnitude2amplitude(self, magnitude, light_model):
        """

        :param magnitude: magnitude of the object
        :param light_model: model type of the object
        :return: value of the lenstronomy 'amp' parameter such that the total flux of the profile type results in this
        magnitude
        """
        amp = 0
        return amp
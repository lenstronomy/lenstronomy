from lenstronomy.SimulationAPI.observation_type import SingleBand


class SimAPI(SingleBand):
    """
    This class manages the model parameters in regard of the data specified in SingleBand. In particular,
    this API translates models specified in units of astronomical magnitudes into the amplitude parameters used in the
    LightModel module of lenstronomy.
    Optionally, this class can also handle inputs with cosmology dependent lensing quantities and translates them to
    the optical quantities being used in the lenstronomy LensModel module.
    All other model choices are equivalent to the ones provided by LightModel, LensModel, PointSource modules
    """
    def __init__(self, kwargs_single_band):
        SingleBand.__init__(**kwargs_single_band)

    def magnitude2amplitude(self, magnitude, light_model):
        """

        :param magnitude: magnitude of the object
        :param light_model: model type of the object
        :return: value of the lenstronomy 'amp' parameter such that the total flux of the profile type results in this
        magnitude
        """
        amp = 0
        return amp
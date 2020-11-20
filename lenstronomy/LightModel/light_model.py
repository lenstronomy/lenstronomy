__author__ = 'sibirrer'


from lenstronomy.LightModel.linear_basis import LinearBasis

__all__ = ['LightModel']


class LightModel(LinearBasis):
    """
    class to handle extended surface brightness profiels (for e.g. source and lens light)

    all profiles come with a surface_brightness parameterization (in units per square angle and independent of
    the pixel scale).
    The parameter 'amp' is the linear scaling parameter of surface brightness.
    Some functional forms come with a total_flux() definition that provide the integral of the surface brightness for a
    given set of parameters.

    The SimulationAPI module allows to use astronomical magnitudes to be used and translated into the surface brightness
    conventions of this module given a magnitude zero point.

    """

    def __init__(self, light_model_list, deflection_scaling_list=None, source_redshift_list=None,
                 smoothing=0.001):
        super(LightModel, self).__init__(light_model_list=light_model_list,
                                         smoothing=smoothing)
        self.deflection_scaling_list = deflection_scaling_list
        self.redshift_list = source_redshift_list

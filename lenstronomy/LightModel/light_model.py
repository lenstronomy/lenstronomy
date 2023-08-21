__author__ = "sibirrer"


from lenstronomy.LightModel.linear_basis import LinearBasis

__all__ = ["LightModel"]


class LightModel(LinearBasis):
    """Class to handle extended surface brightness profiles (for e.g. source and lens
    light)

    all profiles come with a surface_brightness parameterization (in units per square
    angle and independent of the pixel scale). The parameter 'amp' is the linear scaling
    parameter of surface brightness. Some functional forms come with a total_flux()
    definition that provide the integral of the surface brightness for a given set of
    parameters.

    The SimulationAPI module allows to use astronomical magnitudes to be used and
    translated into the surface brightness conventions of this module given a magnitude
    zero point.
    """

    def __init__(
        self,
        light_model_list,
        deflection_scaling_list=None,
        source_redshift_list=None,
        smoothing=0.001,
        sersic_major_axis=None,
    ):
        """

        :param light_model_list: list of light models
        :param deflection_scaling_list: list of floats indicating a relative scaling of the deflection angle from the
         reduced angles in the lens model definition (optional, only possible in single lens plane with multiple source
         planes)
        :param source_redshift_list: list of redshifts for the different light models
         (optional and only used in multi-plane lensing in conjunction with a cosmology model)
        :param smoothing: smoothing factor for certain models (deprecated)
        :param sersic_major_axis: boolean or None, if True, uses the semi-major axis as the definition of the Sersic
         half-light radius, if False, uses the product average of semi-major and semi-minor axis. If None, uses the
         convention in the lenstronomy yaml setting (which by default is =False)
        """
        super(LightModel, self).__init__(
            light_model_list=light_model_list,
            smoothing=smoothing,
            sersic_major_axis=sersic_major_axis,
        )
        self.deflection_scaling_list = deflection_scaling_list
        self.redshift_list = source_redshift_list

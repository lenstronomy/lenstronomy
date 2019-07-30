__author__ = 'sibirrer'

#this file contains a class which describes the surface brightness of the light models

from lenstronomy.LightModel.linear_basis import LinearBasis


class LightModel(LinearBasis):
    """
    class to handle source and lens light models
    """

    def __init__(self, light_model_list, deflection_scaling_list=None, source_redshift_list=None,
                 smoothing=0.0000001):
        super(LightModel, self).__init__(light_model_list=light_model_list,
                                         smoothing=smoothing)
        self.deflection_scaling_list = deflection_scaling_list
        self.redshift_list = source_redshift_list

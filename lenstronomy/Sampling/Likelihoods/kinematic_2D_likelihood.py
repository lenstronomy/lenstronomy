import numpy as np

__all__ = ['KinLikelihood']


class KinLikelihood(object):
    """
    Class to compute hte likelihood associated to binned 2D kinematic maps
    """
    def __init__(self, kinematic_data_2D_class, lens_model_class, lens_light_model_class):
        """
        :param kinematic_data_2D_class: KinData class instance
        :param lens_model_class: LensModel class instance
        :param lens_light_model_class: LightModel class instance
        """
    def logL(self, kwargs_lens, kwargs_lens_light, kwargs_special):
        return 0
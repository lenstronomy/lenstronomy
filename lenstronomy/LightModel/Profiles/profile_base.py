import numpy as np

__all__ = ['LightProfileBase']


class LightProfileBase(object):
    """
    base class of all light profiles
    """
    def __init__(self):
        pass

    def function(self, *args, **kwargs):
        """

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: keyword arguments of profile
        :return: surface brightness, raise as definition is not defined
        """
        raise ValueError('function definition not defined in the light profile.')

    def light_3d(self, *args, **kwargs):
        """

        :param r: 3d radius
        :param kwargs:  keyword arguments of profile
        :return: 3d light profile, raise as definition is not defined
        """
        raise ValueError('light_3d definition not defined in the light profile.')

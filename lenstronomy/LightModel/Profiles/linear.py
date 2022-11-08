import numpy as np
import lenstronomy.Util.param_util as param_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class Linear(object):
    """
    class for two-dimensional Linear light profile

    profile name in LightModel module: 'LINEAR'
    """
    def __init__(self):
        self.param_names = ['amp', 'b', 'center_x', 'center_y']
        self.lower_limit_default = {'b': 0, 'amp': -100, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'b': 1000, 'amp': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, b, center_x=0, center_y=0):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: slope value for the linear profile
        :param b: the intercept value before time amp
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        dis = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 )
        f = amp * (dis + b/amp)
        f[f<0] = 0
        return f

    def total_flux(self, amp, b, center_x=0, center_y=0):
        """
        integrated flux of the profile

        :param amp: slope value for the linear profile
        :param b: the intercept value
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return 1/3*(np.pi * (b/abs(amp)) ** 2 * b)


@export
class LinearEllipse(object):
    """
    class for Linear light profile with ellipticity

    profile name in LightModel module: 'LINEAR_ELLIPSE'
    """
    param_names = ['amp', 'b', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'b': 0, 'amp': -100, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'b': 1000, 'amp': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.linear = Linear()

    def function(self, x, y, amp, b, e1, e2, center_x=0, center_y=0):
        """

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: slope value for the linear profile
        :param b: the intercept value before time amp
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        return self.linear.function(x_, y_, amp, b, center_x=0, center_y=0)

    def total_flux(self, amp, b, e1=None, e2=None, center_x=None, center_y=None):
        """
        total integrated flux of profile

        :param amp: slope value for the linear profile
        :param b: the intercept value
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return self.linear.total_flux(amp, b, center_x, center_y)

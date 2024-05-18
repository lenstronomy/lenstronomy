__author__ = 'martin-millon'

#  this file contains a class to make a thin disk profile (Shakura & Sunyaev 1973)
import numpy as np
from lenstronomy.Util import param_util

__all__ = ['ThinDisk', 'ThinDiskEllipse', 'ThinDiskEccentric']

class ThinDisk(object):
    """
    This class contains functions to evaluate a thin disk model

    .. math::
        I(R) = I_0 / \\left[\\exp(\\xi) - 1\\right]

    with :math:`I_0 = amp`
        and
    with :math:`\\xi (R) = \\left[ R / R_0 \\right] ^{3/4}  \\left[ 1 - \\sqrt{R_in/R}\\right] ^{-1/4}`

    """

    param_names = ['amp', 'R_0', 'R_in', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_0': 0.0, 'R_in': 0.0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100000, 'R_0': 100, 'R_in': 10, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_0, R_in, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_0: scale radius
        :param R_in: inner radius of the profile
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :return: Thin disk profile value at (x, y)
        """
        R = np.sqrt((x - center_x) ** 2 + (y - center_y)**2)
        profile = np.zeros_like(R)
        if R_in <= 0:
            profile = amp / (np.exp((R / R_0) ** (0.75)) - 1)
        else:
            profile = np.where(R <= R_in, profile,
                               amp / (np.exp((R / R_0) ** (0.75) * (1 - np.sqrt(R_in / R)) ** (-0.25)) - 1))

        return profile

class ThinDiskEllipse(object):
    """
    this class contains functions to evaluate an elliptical thin disk model

    """

    param_names = ['amp', 'R_0', 'R_in', 'e1','e2','center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_0': 0.0, 'R_in': 0.0, 'e1': -0.5, 'e2': -0.5,'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100000, 'R_0': 100, 'R_in': 10, 'e1': 0.5, 'e2':-0.5,'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_0, R_in, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_0: scale radius
        :param R_in: inner radius of the profile
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param e1: eccentricity parameter
        :param e2: eccentricity parameter
        :return: Thin disk profile value at (x, y)
        """

        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        R_circular = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        R = np.sqrt((x_) ** 2 + (y_)**2)
        profile = np.zeros_like(R)
        if R_in <= 0:
            profile = amp / (np.exp((R / R_0) ** (0.75)) - 1)
        else:
            profile = np.where(R <= R_in, profile,
                               amp / (np.exp((R / R_0) ** (0.75) * (1 - np.sqrt(R_in / R)) ** (-0.25)) - 1))

        return profile


class ThinDiskEccentric(object):
    """
    this class contains functions to evaluate an eccentric thin disk model (as in Eracleous et al. 1995)

    """

    param_names = ['amp', 'R_0', 'R_in','e', 'phi_0','center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'R_0': 0.0, 'R_in': 0.0, 'e': 0., 'phi_0': 0.,'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100000, 'R_0': 100, 'R_in': 10, 'e': 1.0, 'phi_0':2*np.pi,'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, R_0, R_in, e, phi_0, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp: surface brightness/amplitude value at the half light radius
        :param R_0: scale radius
        :param R_in: inner radius of the profile
        :param center_x: center in x-coordinate
        :param center_y: center in y-coordinate
        :param phi_0: angle : 0 corresponds to rthe apocenter along the x direction
        :param e: eccentricity of the orbit
        :return: Thin disk profile value at (x, y)
        """

        x_, y_ = self.eccentric_coordinate_system(x, y, e, phi_0, center_x, center_y)
        R = np.sqrt((x_) ** 2 + (y_)**2)
        #rescaling of the caracteristique scales
        R_0 = R_0 / (1+e)
        R_in = R_in / (1+e)

        profile = np.zeros_like(R)
        if R_in <= 0:
            profile = amp / (np.exp((R / R_0) ** (0.75)) - 1)
        else:
            profile = np.where(R <= R_in, profile,
                               amp / (np.exp((R / R_0) ** (0.75) * (1 - np.sqrt(R_in / R)) ** (-0.25)) - 1))

        return profile

    def eccentric_coordinate_system(self, x, y, e, phi_0, center_x, center_y):
        """
        maps the coordinates x, y into an eccentric coordinate system

        :param x: x-coordinate
        :param y: y-coordinate
        :param e: eccentricity
        :param center_x: center of distortion
        :param center_y: center of distortion
        :return: distorted coordinates x', y'
        """
        R = np.sqrt((x- center_x)**2 + (y-center_y)**2)
        phi = np.arctan2((y-center_y), (x-center_x))
        a = (R / (1 + e)) * (1 - e * np.cos(phi - phi_0))

        xt1 = a * np.cos(phi)
        xt2 = a * np.sin(phi)
        return xt1, xt2
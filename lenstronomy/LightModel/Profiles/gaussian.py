import numpy as np
import lenstronomy.Util.param_util as param_util

from lenstronomy.Util.package_util import exporter
export, __all__ = exporter()


@export
class Gaussian(object):
    """
    class for Gaussian light profile
    The two-dimensional Gaussian profile amplitude is defined such that the 2D integral leads to the 'amp' value.

    profile name in LightModel module: 'GAUSSIAN'
    """
    def __init__(self):
        self.param_names = ['amp', 'sigma', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'sigma': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 1000, 'sigma': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        c = amp / (2 * np.pi * sigma**2)
        R2 = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        return c * np.exp(-R2 / 2.)

    def total_flux(self, amp, sigma, center_x=0, center_y=0):
        """
        integrated flux of the profile

        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return amp

    def light_3d(self, r, amp, sigma):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :return: 3D brightness per angular volume element
        """
        amp3d = amp / np.sqrt(2 * sigma**2) / np.sqrt(np.pi)
        sigma3d = sigma
        return self.function(r, 0, amp3d, sigma3d)


@export
class GaussianEllipse(object):
    """
    class for Gaussian light profile with ellipticity

    profile name in LightModel module: 'GAUSSIAN_ELLIPSE'
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        return self.gaussian.function(x_, y_, amp, sigma, center_x=0, center_y=0)

    def total_flux(self, amp, sigma=None, e1=None, e2=None, center_x=None, center_y=None):
        """
        total integrated flux of profile

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return self.gaussian.total_flux(amp, sigma, center_x, center_y)

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: amplitude, such that 2D integral leads to this value
        :param sigma: sigma of Gaussian in each direction
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: 3D brightness per angular volume element
        """
        return self.gaussian.light_3d(r, amp, sigma=sigma)


@export
class MultiGaussian(object):
    """
    class for elliptical pseudo Jaffe lens light (2d projected light/mass distribution

    profile name in LightModel module: 'MULTI_GAUSSIAN'
    """
    param_names = ['amp', 'sigma', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
        return f_

    def total_flux(self, amp, sigma, center_x=0, center_y=0):
        """
        total integrated flux of profile

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        flux = 0
        for i in range(len(amp)):
            flux += self.gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    def function_split(self, x, y, amp, sigma, center_x=0, center_y=0):
        """
        split surface brightness in individual components

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param center_x: center of profile
        :param center_y: center of profile
        :return: list of arrays of surface brightness
        """
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y))
        return f_list

    def light_3d(self, r, amp, sigma):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :return: 3D brightness per angular volume element
        """
        f_ = np.zeros_like(r)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_


@export
class MultiGaussianEllipse(object):
    """
    class for elliptical multi Gaussian profile

    profile name in LightModel module: 'MULTI_GAUSSIAN_ELLIPSE'
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        surface brightness per angular unit

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)

        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0)
        return f_

    def total_flux(self, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        total integrated flux of profile

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        flux = 0
        for i in range(len(amp)):
            flux += self.gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    def function_split(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """
        split surface brightness in individual components

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :param center_x: center of profile
        :param center_y: center of profile
        :return: list of arrays of surface brightness
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0))
        return f_list

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """
        3D brightness per angular volume element

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma: list of widths of individual Gaussian profiles
        :param e1: eccentricity modulus
        :param e2: eccentricity modulus
        :return: 3D brightness per angular volume element
        """
        f_ = np.zeros_like(r)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_

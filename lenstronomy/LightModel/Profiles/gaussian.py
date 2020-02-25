import numpy as np
import lenstronomy.Util.param_util as param_util


class Gaussian(object):
    """
    class for Gaussian light profile
    """
    def __init__(self):
        self.param_names = ['amp', 'sigma', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'sigma': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 1000, 'sigma': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp: amplitude
        :param sigma: sigma of Gaussian
        :param center_x:
        :param center_y:
        :return:
        """
        c = amp / (2 * np.pi * sigma**2)
        R2 = (x - center_x) ** 2 / sigma**2 + (y - center_y) ** 2 / sigma**2
        return c * np.exp(-R2 / 2.)

    def total_flux(self, amp, sigma, center_x=0, center_y=0):
        """

        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        return amp

    def light_3d(self, r, amp, sigma):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        amp3d = amp / np.sqrt(2 * sigma**2) / np.sqrt(np.pi)
        sigma3d = sigma
        return self.function(r, 0, amp3d, sigma3d)


class GaussianEllipse(object):
    """
    class for Gaussian light profile
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        return self.gaussian.function(x_, y_, amp, sigma, center_x=0, center_y=0)

    def total_flux(self, amp, sigma=None, e1=None, e2=None, center_x=None, center_y=None):
        """

        :param amp:
        :param sigma:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.gaussian.total_flux(amp, sigma, center_x, center_y)

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.gaussian.light_3d(r, amp, sigma=sigma)


class MultiGaussian(object):
    """
    class for elliptical pseudo Jaffe lens light (2d projected light/mass distribution
    """
    param_names = ['amp', 'sigma', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
        return f_

    def total_flux(self, amp, sigma, center_x=0, center_y=0):
        """

        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        flux = 0
        for i in range(len(amp)):
            flux += self.gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    def function_split(self, x, y, amp, sigma, center_x=0, center_y=0):
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y))
        return f_list

    def light_3d(self, r, amp, sigma):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(r)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_


class MultiGaussianEllipse(object):
    """
    class for elliptical multi Gaussian profile
    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 1000, 'sigma': 100, 'e1': -0.5, 'e2': -0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian = Gaussian()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)

        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0)
        return f_

    def total_flux(self, amp, sigma, e1, e2, center_x=0, center_y=0):
        """

        :param amp:
        :param sigma:
        :param e1:
        :param e2:
        :param center_x:
        :param center_y:
        :return:
        """
        flux = 0
        for i in range(len(amp)):
            flux += self.gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    def function_split(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        x_, y_ = param_util.transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x_, y_, amp[i], sigma[i], center_x=0, center_y=0))
        return f_list

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(r)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_

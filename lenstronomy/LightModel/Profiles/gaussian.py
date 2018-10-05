import numpy as np
import lenstronomy.Util.param_util as param_util


class Gaussian(object):
    """
    class for Gaussian light profile
    """
    def __init__(self):
        self.param_names = ['amp', 'sigma_x', 'sigma_y', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'sigma_x': 0, 'sigma_y': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 1000, 'sigma_x': 100, 'sigma_y': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
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
        c = amp / (2 * np.pi * sigma_x * sigma_y)
        R2 = (x - center_x) ** 2/sigma_x**2 + (y - center_y) ** 2/sigma_y**2
        return c * np.exp(-R2 / 2.)

    def light_3d(self, r, amp, sigma_x, sigma_y):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        amp3d = amp / np.sqrt(2* sigma_x * sigma_y) / np.sqrt(np.pi)
        sigma3d_x = sigma_x
        sigma3d_y = sigma_y
        return self.function(r, 0, amp3d, sigma3d_x, sigma3d_y)


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
        x_, y_ = param_util.transform_e1e2(x, y, e1, e2)
        return self.gaussian.function(x_, y_, amp, sigma, sigma, center_x, center_y)

    def light_3d(self, r, amp, sigma, e1=0, e2=0):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.gaussian.light_3d(r, amp, sigma_x=sigma, sigma_y=sigma)


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
            f_ += self.gaussian.function(x, y, amp[i], sigma[i], sigma[i], center_x, center_y)
        return f_

    def function_split(self, x, y, amp, sigma, center_x=0, center_y=0):
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x, y, amp[i], sigma[i], sigma[i], center_x, center_y))
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
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i], sigma[i])
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
        x_, y_ = param_util.transform_e1e2(x, y, e1, e2)

        f_ = np.zeros_like(x)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x_, y_, amp[i], sigma[i], sigma[i], center_x, center_y)
        return f_

    def function_split(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0):
        x_, y_ = param_util.transform_e1e2(x, y, e1, e2)
        f_list = []
        for i in range(len(amp)):
            f_list.append(self.gaussian.function(x_, y_, amp[i], sigma[i], sigma[i], center_x, center_y))
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
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i], sigma[i])
        return f_


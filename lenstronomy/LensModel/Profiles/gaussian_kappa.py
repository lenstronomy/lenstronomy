__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np
import scipy.special
import scipy.integrate as integrate
from lenstronomy.LensModel.Profiles.gaussian import Gaussian

class GaussianKappa(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    """
    def __init__(self):
        self.gaussian = Gaussian()
        self.ds = 0.00001

    def function(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        c = amp/(2*np.pi*sigma_x*sigma_y)
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        c = 1. / (2 * sigma_x * sigma_y)
        num_int = self._num_integral(r, c)
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        amp2d = amp_density / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))
        amp2d *= 2 * 1. / (2 * c)
        return num_int * amp2d

    def _num_integral(self, r, c):
        """
        numerical integral (1-e^{-c*x^2})/x dx [0..r]
        :param r: radius
        :param c: 1/2sigma^2
        :return:
        """
        out = integrate.quad(lambda x: (1-np.exp(-c*x**2))/x, 0, r)
        return out[0]

    def derivatives(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_ = x - center_x
        y_ = y - center_y
        R = np.sqrt(x_**2 + y_**2)
        if isinstance(R, int) or isinstance(R, float):
            R = max(R, self.ds)
        else:
            R[R <= self.ds] = self.ds
        alpha = self.alpha_abs(R, amp, sigma_x, sigma_y)
        return alpha / R * x_, alpha / R * y_

    def hessian(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_ = x - center_x
        y_ = y - center_y
        r = np.sqrt(x_**2 + y_**2)
        if isinstance(r, int) or isinstance(r, float):
            r = max(r, self.ds)
        else:
            r[r <= self.ds] = self.ds
        d_alpha_dr = -self.d_alpha_dr(r, amp, sigma_x, sigma_y)
        alpha = self.alpha_abs(r, amp, sigma_x, sigma_y)

        f_xx = -(d_alpha_dr/r + alpha/r**2) * x_**2/r + alpha/r
        f_yy = -(d_alpha_dr/r + alpha/r**2) * y_**2/r + alpha/r
        f_xy = -(d_alpha_dr/r + alpha/r**2) * x_*y_/r
        return f_xx, f_yy, f_xy

    def density(self, r, amp, sigma_x, sigma_y):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        return self.gaussian.function(r, 0, amp, sigma_x, sigma_y)

    def density_2d(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        amp2d = self._amp3d_to_2d(amp, sigma_x, sigma_y)
        return self.gaussian.function(x, y, amp2d, sigma_x, sigma_y, center_x, center_y)

    def mass_2d(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        amp2d = amp / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))
        c = 1./(2 * sigma_x * sigma_y)
        return amp2d * 2 * np.pi * 1./(2*c) * (1. - np.exp(-c * R**2))

    def mass_2d_lens(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_2d(R, amp_density, sigma_x, sigma_y)

    def alpha_abs(self, R, amp, sigma_x, sigma_y):
        """
        absolute value of the deflection
        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        alpha = self.mass_2d(R, amp_density, sigma_x, sigma_y) / np.pi / R
        return alpha

    def d_alpha_dr(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        c = 1. / (2 * sigma_x * sigma_y)
        A = self._amp2d_to_3d(amp, sigma_x, sigma_y) * np.sqrt(2/np.pi*sigma_x*sigma_y)
        return 1./R**2 * (-1 + (1 + 2*c*R**2) * np.exp(-c*R**2)) * A

    def mass_3d(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        A = amp / (2 * np.pi * sigma_x * sigma_y)
        c = 1. / (2 * sigma_x * sigma_y)
        result = 1. /(2*c) * (-R * np.exp(-c*R**2) + scipy.special.erf(np.sqrt(c) * R) * np.sqrt(np.pi/(4 * c)))
        return result*A * 4 * np.pi

    def mass_3d_lens(self, R, amp, sigma_x, sigma_y):
        """

        :param R:
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        amp_density = self._amp2d_to_3d(amp, sigma_x, sigma_y)
        return self.mass_3d(R, amp_density, sigma_x, sigma_y)

    def _amp3d_to_2d(self, amp, sigma_x, sigma_y):
        """
        converts 3d density into 2d density parameter
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return amp * np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2)

    def _amp2d_to_3d(self, amp, sigma_x, sigma_y):
        """
        converts 3d density into 2d density parameter
        :param amp:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        return amp / (np.sqrt(np.pi) * np.sqrt(sigma_x * sigma_y * 2))



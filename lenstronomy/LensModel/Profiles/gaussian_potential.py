__author__ = 'sibirrer'
#this file contains a class to make a gaussian

import numpy as np


class Gaussian(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    """
    param_names = ['amp', 'sigma_x', 'sigma_y', 'center_x', 'center_y']

    def function(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """
        returns Gaussian
        """
        c = amp/(2*np.pi*sigma_x*sigma_y)
        delta_x = x - center_x
        delta_y = y - center_y
        exponent = -((delta_x/sigma_x)**2+(delta_y/sigma_y)**2)/2.
        return c * np.exp(exponent)

    def derivatives(self, x, y, amp, sigma_x, sigma_y, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        return f_ * (center_x-x)/sigma_x**2, f_ * (center_y-y)/sigma_y**2

    def hessian(self, x, y, amp, sigma_x, sigma_y, center_x = 0, center_y = 0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        f_ = self.function(x, y, amp, sigma_x, sigma_y, center_x, center_y)
        f_xx = f_ * ( (-1./sigma_x**2) + (center_x-x)**2/sigma_x**4 )
        f_yy = f_ * ( (-1./sigma_y**2) + (center_y-y)**2/sigma_y**4 )
        f_xy = f_ * (center_x-x)/sigma_x**2 * (center_y-y)/sigma_y**2
        return f_xx, f_yy, f_xy

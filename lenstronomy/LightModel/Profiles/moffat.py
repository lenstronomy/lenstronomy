__author__ = 'sibirrer'

#this file contains a class to make a moffat profile


class Moffat(object):
    """
    this class contains functions to evaluate a Gaussian function and calculates its derivative and hessian matrix
    """
    def __init__(self):
        self.param_names = ['amp', 'alpha', 'beta', 'center_x', 'center_y']

    def function(self, x, y, amp, alpha, beta, center_x, center_y):
        """
        returns Moffat profile
        """
        x_shift = x - center_x
        y_shift = y - center_y
        return amp * (1. + (x_shift**2+y_shift**2)/alpha**2)**(-beta)

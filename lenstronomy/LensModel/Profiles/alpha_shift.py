__author__ = 'sibirrer'

import lenstronomy.Util.param_util as param_util
import numpy as np


class Shift(object):
    """
    new class for external shear e1, e2 expression
    """
    param_names = ['alpha_x', 'alpha_y']
    lower_limit_default = {'alpha_x': -1000, 'alpha_y': -1000}
    upper_limit_default = {'alpha_x': 1000, 'alpha_y': 1000}

    def function(self, x, y, alpha_x, alpha_y):

        return np.zeros_like(x)

    def derivatives(self, x, y, alpha_x, alpha_y):
        f_x = np.ones_like(x) * alpha_x
        f_y = np.ones_like(x) * alpha_y
        return f_x, f_y

    def hessian(self, x, y, alpha_x, alpha_y):
        f_xx = 0
        f_yy = 0
        f_xy = 0
        return f_xx, f_yy, f_xy

__author__ = 'sibirrer'

import numpy as np

class SIS(object):
    """
    this class contains the function and the derivatives of the Singular Isothermal Sphere
    """
    param_names = ['theta_E', 'center_x', 'center_y']

    def function(self, x, y, theta_E, center_x=0, center_y=0):
        x_shift = x - center_x
        y_shift = y - center_y
        f_ = theta_E * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        return f_

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            a = theta_E / max(0.000001, R)
        else:
            a=np.empty_like(R)
            r = R[R > 0]  #in the SIS regime
            a[R == 0] = 0
            a[R > 0] = theta_E / r
        f_x = a * x_shift
        f_y = a * y_shift
        return f_x, f_y

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        if isinstance(R, int) or isinstance(R, float):
            prefac = theta_E / max(0.000001, R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = theta_E / r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_xx, f_yy, f_xy
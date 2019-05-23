__author__ = 'dgilman'

from scipy.integrate import quad
import numpy as np

class EXPDISK(object):
    """
    this class contains the function and the derivatives of an elliptical exponential profile
    with the ellipticity included in the convergence (not the potential)
    """
    param_names = ['k_eff', 'R_eff', 'n', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'k_eff': 0, 'R_eff': 0, 'n': 0.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100,
                           'center_y': -100}
    upper_limit_default = {'k_eff': 10, 'R_eff': 100, 'n': 8, 'e1': 0.5, 'e2': 0.5, 'center_x': 100,
                           'center_y': 100}

    def function(self, x, y, n, R_eff, k_eff, e1, e2, center_x=0, center_y=0):
        raise Exception('not yet implemented')
        return f_

    def _coord_transf(self, x, y, q, phi_G, center_x, center_y):
        """

        :param x:
        :param y:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        x_shift = x - center_x
        y_shift = y - center_y
        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)
        e = abs(1 - q)
        x_ = (cos_phi * x_shift + sin_phi * y_shift) * np.sqrt(1 - e)
        y_ = (-sin_phi * x_shift + cos_phi * y_shift) * np.sqrt(1 + e)
        return x_, y_

    def _kappa(self, x, y, n, R_eff, k_eff, q, phi_G, center_x = 0, center_y = 0):

        _x, _y = self._coord_transf(x, y, q, phi_G, center_x, center_y)

        R = (_x**2 + _y**2) ** 0.5

        exponent = - (R * R_eff**-1) ** (-n**-1)

        return k_eff * np.exp(exponent)

    def _integrand_deflection(self, x, y, n, R_eff, k_eff, q, phi_G, center_x, center_y, R_max):

        integral = quad(self._kappa, 0, R_max)

    def derivatives(self, x, y, n, R_eff, k_eff, e1, e2, center_x=0, center_y=0):
        """
        returns df/dx and df/dy of the function
        """
        x_shift = x - center_x
        y_shift = y - center_y
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)




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
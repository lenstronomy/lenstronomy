__author__ = 'sibirrer'


import numpy as np
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ['PointMass']


class PointMass(LensProfileBase):
    """
    class to compute the physical deflection angle of a point mass, given as an Einstein radius
    """
    param_names = ['theta_E', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.r_min = 10**(-25)
        super(PointMass, self).__init__()
        # alpha = 4*const.G * (mass*const.M_sun)/const.c**2/(r*const.Mpc)

    def function(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: lensing potential
        """
        x_ = x - center_x
        y_ = y - center_y
        a = np.sqrt(x_**2 + y_**2)
        if isinstance(a, int) or isinstance(a, float):
            r = max(self.r_min, a)
        else:
            r = np.empty_like(a)
            r[a > self.r_min] = a[a > self.r_min]  #in the SIS regime
            r[a <= self.r_min] = self.r_min
        phi = theta_E**2*np.log(r)
        return phi

    def derivatives(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: deflection angle (in angles)
        """
        x_ = x - center_x
        y_ = y - center_y
        a = np.sqrt(x_**2 + y_**2)
        if isinstance(a, int) or isinstance(a, float):
            r = max(self.r_min, a)
        else:
            r = np.empty_like(a)
            r[a > self.r_min] = a[a > self.r_min]  #in the SIS regime
            r[a <= self.r_min] = self.r_min
        alpha = theta_E**2/r
        return alpha*x_/r, alpha*y_/r

    def hessian(self, x, y, theta_E, center_x=0, center_y=0):
        """

        :param x: x-coord (in angles)
        :param y: y-coord (in angles)
        :param theta_E: Einstein radius (in angles)
        :return: hessian matrix (in angles)
        """
        x_ = x - center_x
        y_ = y - center_y
        C = theta_E**2
        a = x_**2 + y_**2
        if isinstance(a, int) or isinstance(a, float):
            r2 = max(self.r_min**2, a)
        else:
            r2 = np.empty_like(a)
            r2[a > self.r_min**2] = a[a > self.r_min**2]  #in the SIS regime
            r2[a <= self.r_min**2] = self.r_min**2
        f_xx = C * (y_**2-x_**2)/r2**2
        f_yy = C * (x_**2-y_**2)/r2**2
        f_xy = -C * 2*x_*y_/r2**2
        return f_xx, f_xy, f_xy, f_yy

__author__ = 'maverickoh'

import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.epl import EPL
from lenstronomy.LensModel.Profiles.multipole import Multipole

__all__ = ['EPL_BOXYDISKY']

class EPL_BOXYDISKY(LensProfileBase):
    """"
    EPL (Elliptical Power Law) mass profile, combined with MULTIPOLE with m=4
    It adds an aligned boxy/diskyness (MULTIPOLE, m=4)
    """
    param_names = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y', 'a_m']
    lower_limit_default = {'theta_E': 0, 'gamma': 1.5, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100,\
                           'a_m': -0.1}
    upper_limit_default = {'theta_E': 100, 'gamma': 2.5, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100,\
                           'a_m': +0.1}

    def __init__(self):
        self.epl = EPL()
        # self.epl_major_axis = EPLMajorAxis()
        self.multipole = Multipole()
        self.m = int(4)
        super(EPL_BOXYDISKY, self).__init__()

    def set_static(self, theta_E, gamma, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: self variables set
        """
        self._static = True
        self._b_static, self._t_static, self._q_static, self._phi_G_static = self.epl._param_conv(theta_E, gamma, e1,
                                                                                                  e2)

    def set_dynamic(self):
        """

        :return:
        """
        self._static = False
        if hasattr(self, '_b_static'):
            del self._b_static
        if hasattr(self, '_t_static'):
            del self._t_static
        if hasattr(self, '_phi_G_static'):
            del self._phi_G_static
        if hasattr(self, '_q_static'):
            del self._q_static

    def function(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: lensing potential
        """
        _, _, _, phi = self.epl.param_conv(theta_E, gamma, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        f_epl = self.epl.function(self, x_, y_, theta_E, gamma, e1, e2)
        f_multipole = self.multipole.function(self, x_, y_, self.m, a_m, phi)
        return f_epl + f_multipole

    def derivatives(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: alpha_x, alpha_y
        """
        _, _, _, phi = self.epl.param_conv(theta_E, gamma, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        f_x_epl, f_y_epl = self.epl.derivatives(x_, y_, theta_E, gamma, e1, e2)
        f_x_multipole, f_y_multipole = self.multipole.derivatives(x_, y_, self.m, a_m, phi)
        f_x = f_x_epl + f_x_multipole
        f_y = f_y_epl + f_y_multipole
        return f_x, f_y

    def hessian(self, x, y, theta_E, gamma, e1, e2, a_m, center_x=0, center_y=0):
        """

        :param x: x-coordinate in image plane
        :param y: y-coordinate in image plane
        :param theta_E: Einstein radius
        :param gamma: power law slope
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: profile center
        :param center_y: profile center
        :return: f_xx, f_xy, f_yx, f_yy
        """
        _, _, _, phi = self.epl.param_conv(theta_E, gamma, e1, e2)
        x_ = x - center_x
        y_ = y - center_y
        f_xx_epl, f_xy_epl, f_yx_epl, f_yy_epl = self.epl.hessian(self, x_, y_, theta_E, gamma, e1, e2)
        f_xx_multipole, f_xy_multipole, f_yx_multipole, f_yy_multipole = self.multipole.hessian(x_, y_, self.m,
                                                                                                    a_m, phi)
        f_xx = f_xx_epl + f_xx_multipole
        f_xy = f_xy_epl = f_xy_multipole
        f_yx = f_yx_epl + f_yx_multipole
        f_yy = f_yy_epl + f_yy_multipole
        return f_xx, f_xy, f_yx, f_yy

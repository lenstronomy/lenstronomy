from lenstronomy.LensModel.Profiles.nie import NIE
import lenstronomy.Util.param_util as param_util
import numpy as np


class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['theta_E', 'w_c', 'w_t', 'e1', 'e2', 'center_x', 'center_y']

    def __init__(self):
        self.nie = NIE()

    def function(self, x, y, theta_E, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param amp: amplitude of first power-law flux
        :param flux_ratio: ratio of amplitudes of first to second power-law profile
        :param gamma1: power-law slope
        :param gamma2: power-law slope
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: center
        :param center_y: center
        :return: flux of chameleon profile
        """
        if not w_t > w_c:
            w_t, w_c = w_c, w_t
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_1 = self.nie.function(x, y, theta_E, e1, e2, s_scale_1, center_x, center_y)
        f_2 = self.nie.function(x, y, theta_E, e1, e2, s_scale_2, center_x, center_y)
        f_ = f_1 - f_2
        return f_

    def derivatives(self, x, y, theta_E, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param amp: amplitude of first power-law flux
        :param flux_ratio: ratio of amplitudes of first to second power-law profile
        :param gamma1: power-law slope
        :param gamma2: power-law slope
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: center
        :param center_y: center
        :return: flux of chameleon profile
        """
        if not w_t > w_c:
            w_t, w_c = w_c, w_t
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_x_1, f_y_1 = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale_1, center_x, center_y)
        f_x_2, f_y_2 = self.nie.derivatives(x, y, theta_E, e1, e2, s_scale_2, center_x, center_y)
        f_x = f_x_1 - f_x_2
        f_y = f_y_1 - f_y_2
        return f_x, f_y

    def hessian(self, x, y, theta_E, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param theta_E: amplitude of first power-law flux
        :param flux_ratio: ratio of amplitudes of first to second power-law profile
        :param gamma1: power-law slope
        :param gamma2: power-law slope
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: center
        :param center_y: center
        :return: flux of chameleon profile
        """
        if not w_t > w_c:
            w_t, w_c = w_c, w_t
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_xx_1, f_yy_1, f_xy_1 = self.nie.hessian(x, y, theta_E, e1, e2, s_scale_1, center_x, center_y)
        f_xx_2, f_yy_2, f_xy_2 = self.nie.hessian(x, y, theta_E, e1, e2, s_scale_2, center_x, center_y)
        f_xx = f_xx_1 - f_xx_2
        f_yy = f_yy_1 - f_yy_2
        f_xy = f_xy_1 - f_xy_2
        return f_xx, f_yy, f_xy
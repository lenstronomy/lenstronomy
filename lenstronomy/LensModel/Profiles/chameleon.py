from lenstronomy.LensModel.Profiles.nie import NIE
from lenstronomy.LensModel.Profiles.point_mass import PointMass
import lenstronomy.Util.param_util as param_util
import numpy as np


class Chameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['alpha_1', 'w_c', 'w_t', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'alpha_1': 0, 'w_c': 0, 'w_t': 0, 'e1': -0.8, 'e2': -0.8, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'alpha_1': 100, 'w_c': 100, 'w_t': 100, 'e1': 0.8, 'e2': 0.8, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.nie = NIE()

    def function(self, x, y, alpha_1, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param w_c: see Suyu+2014
        :param w_t: see Suyu+2014
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: ra center
        :param center_y: dec center
        :return: lensing potential
        """

        theta_E_conv, w_c, w_t = self._theta_convert(alpha_1, w_c, w_t)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_1 = self.nie.function(x, y, theta_E_conv, e1, e2, s_scale_1, center_x, center_y)
        f_2 = self.nie.function(x, y, theta_E_conv, e1, e2, s_scale_2, center_x, center_y)
        f_ = f_1 - f_2
        return f_

    def derivatives(self, x, y, alpha_1, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param w_c: see Suyu+2014
        :param w_t: see Suyu+2014
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: ra center
        :param center_y: dec center
        :return: deflection angles (RA, DEC)
        """
        theta_E_conv, w_c, w_t = self._theta_convert(alpha_1, w_c, w_t)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_x_1, f_y_1 = self.nie.derivatives(x, y, theta_E_conv, e1, e2, s_scale_1, center_x, center_y)
        f_x_2, f_y_2 = self.nie.derivatives(x, y, theta_E_conv, e1, e2, s_scale_2, center_x, center_y)
        f_x = f_x_1 - f_x_2
        f_y = f_y_1 - f_y_2
        return f_x, f_y

    def hessian(self, x, y, alpha_1, w_c, w_t, e1, e2, center_x=0, center_y=0):
        """

        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param w_c: see Suyu+2014
        :param w_t: see Suyu+2014
        :param e1: ellipticity parameter
        :param e2: ellipticity parameter
        :param center_x: ra center
        :param center_y: dec center
        :return: second derivatives of the lensing potential (Hessian: f_xx, f_yy, f_xy)
        """
        theta_E_conv, w_c, w_t = self._theta_convert(alpha_1, w_c, w_t)
        phi_G, q = param_util.ellipticity2phi_q(e1, e2)
        s_scale_1 = np.sqrt(4 * w_c ** 2 / (1. + q) ** 2)
        s_scale_2 = np.sqrt(4 * w_t ** 2 / (1. + q) ** 2)
        f_xx_1, f_yy_1, f_xy_1 = self.nie.hessian(x, y, theta_E_conv, e1, e2, s_scale_1, center_x, center_y)
        f_xx_2, f_yy_2, f_xy_2 = self.nie.hessian(x, y, theta_E_conv, e1, e2, s_scale_2, center_x, center_y)
        f_xx = f_xx_1 - f_xx_2
        f_yy = f_yy_1 - f_yy_2
        f_xy = f_xy_1 - f_xy_2
        return f_xx, f_yy, f_xy

    def _theta_convert(self, alpha_1, w_c, w_t):
        """
        convert the parameter alpha_1 (deflection angle one arcsecond from the center) into the
        "Einstein radius" scale parameter of the two NIE profiles

        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param w_c: see Suyu+2014
        :param w_t: see Suyu+2014
        :return:
        """
        if not w_t > w_c:
            w_t, w_c = w_c, w_t
        s_scale_1 = w_c
        s_scale_2 = w_t
        f_x_1, f_y_1 = self.nie.derivatives(1, 0, theta_E=1, e1=0, e2=0, s_scale=s_scale_1)
        f_x_2, f_y_2 = self.nie.derivatives(1, 0, theta_E=1, e1=0, e2=0, s_scale=s_scale_2)
        f_x = f_x_1 - f_x_2
        theta_E_convert = alpha_1 / f_x
        return theta_E_convert, w_c, w_t


class DoubleChameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['alpha_1', 'ratio', 'w_c1', 'w_t1', 'e11', 'e21', 'w_c2', 'w_t2', 'e12', 'e22', 'center_x', 'center_y']
    lower_limit_default = {'alpha_1': 0, 'ratio': 0, 'w_c1': 0, 'w_t1': 0, 'e11': -0.8, 'e21': -0.8,
                           'w_c2': 0, 'w_t2': 0, 'e12': -0.8, 'e22': -0.8,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'alpha_1': 100, 'ratio': 100, 'w_c1': 100, 'w_t1': 100, 'e11': 0.8, 'e21': 0.8,
                           'w_c2': 100, 'w_t2': 100, 'e12': 0.8, 'e22': 0.8,
                           'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.chameleon = Chameleon()

    def function(self, x, y, alpha_1, ratio, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, center_x=0, center_y=0):
        """
        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param ratio: ratio of deflection amplitude at radius = 1 of the first to second Chameleon profile
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return: lensing potential
        """

        f_1 = self.chameleon.function(x, y, alpha_1 / (1. + 1. / ratio), w_c1, w_t1, e11, e21, center_x, center_y)
        f_2 = self.chameleon.function(x, y, alpha_1 / (1. + ratio), w_c2, w_t2, e12, e22, center_x, center_y)
        return f_1 + f_2

    def derivatives(self, x, y, alpha_1, ratio, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, center_x=0, center_y=0):
        """
        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param ratio: ratio of deflection amplitude at radius = 1 of the first to second Chameleon profile
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return: deflection angles (RA, DEC)
        """
        f_x1, f_y1 = self.chameleon.derivatives(x, y, alpha_1 / (1. + 1. / ratio), w_c1, w_t1, e11, e21, center_x, center_y)
        f_x2, f_y2 = self.chameleon.derivatives(x, y, alpha_1 / (1. + ratio), w_c2, w_t2, e12, e22, center_x, center_y)
        return f_x1 + f_x2, f_y1 + f_y2

    def hessian(self, x, y, alpha_1, ratio, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, center_x=0, center_y=0):
        """
        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param ratio: ratio of deflection amplitude at radius = 1 of the first to second Chameleon profile
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return: second derivatives of the lensing potential (Hessian: f_xx, f_yy, f_xy)
        """
        f_xx1, f_yy1, f_xy1 = self.chameleon.hessian(x, y, alpha_1 / (1. + 1. / ratio), w_c1, w_t1, e11, e21, center_x, center_y)
        f_xx2, f_yy2, f_xy2 = self.chameleon.hessian(x, y, alpha_1 / (1. + ratio), w_c2, w_t2, e12, e22, center_x, center_y)
        return f_xx1 + f_xx2, f_yy1 + f_yy2, f_xy1 + f_xy2


class TripleChameleon(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['alpha_1', 'ratio12', 'ratio13', 'w_c1', 'w_t1', 'e11', 'e21', 'w_c2', 'w_t2', 'e12', 'e22', 'w_c3', 'w_t3', 'e13',
                   'e23', 'center_x', 'center_y']
    lower_limit_default = {'alpha_1': 0, 'ratio12': 0, 'ratio13': 0, 'w_c1': 0, 'w_t1': 0, 'e11': -0.8, 'e21': -0.8,
                           'w_c2': 0, 'w_t2': 0, 'e12': -0.8, 'e22': -0.8,
                           'w_c3': 0, 'w_t3': 0, 'e13': -0.8, 'e23': -0.8,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'alpha_1': 100, 'ratio12': 100, 'ratio13': 100, 'w_c1': 100, 'w_t1': 100, 'e11': 0.8, 'e21': 0.8,
                           'w_c2': 100, 'w_t2': 100, 'e12': 0.8, 'e22': 0.8,
                           'w_c3': 100, 'w_t3': 100, 'e13': 0.8, 'e23': 0.8,
                           'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.chameleon = Chameleon()

    def function(self, x, y, alpha_1, ratio12, ratio13, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, w_c3, w_t3, e13, e23,
                 center_x=0, center_y=0):
        """

        :param alpha_1:
        :param ratio12: ratio of first to second amplitude
        :param ratio13: ratio of first to third amplidute
        :param w_c1:
        :param w_t1:
        :param e11:
        :param e21:
        :param w_c2:
        :param w_t2:
        :param e12:
        :param e22:
        :param center_x:
        :param center_y:
        :return:
        """
        amp1 = alpha_1 / (1. + 1. / ratio12 + 1. / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        f_1 = self.chameleon.function(x, y, amp1, w_c1, w_t1, e11, e21, center_x, center_y)
        f_2 = self.chameleon.function(x, y, amp2, w_c2, w_t2, e12, e22, center_x, center_y)
        f_3 = self.chameleon.function(x, y, amp3, w_c3, w_t3, e13, e23, center_x, center_y)
        return f_1 + f_2 + f_3

    def derivatives(self, x, y, alpha_1, ratio12, ratio13, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, w_c3, w_t3, e13, e23,
                 center_x=0, center_y=0):
        """

        :param alpha_1:
        :param ratio12: ratio of first to second amplitude
        :param ratio13: ratio of first to third amplidute
        :param w_c1:
        :param w_t1:
        :param e11:
        :param e21:
        :param w_c2:
        :param w_t2:
        :param e12:
        :param e22:
        :param center_x:
        :param center_y:
        :return:
        """
        amp1 = alpha_1 / (1. + 1. / ratio12 + 1. / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        f_x1, f_y1 = self.chameleon.derivatives(x, y, amp1, w_c1, w_t1, e11, e21, center_x, center_y)
        f_x2, f_y2 = self.chameleon.derivatives(x, y, amp2, w_c2, w_t2, e12, e22, center_x, center_y)
        f_x3, f_y3 = self.chameleon.derivatives(x, y, amp3, w_c3, w_t3, e13, e23, center_x, center_y)
        return f_x1 + f_x2 + f_x3, f_y1 + f_y2 + f_y3

    def hessian(self, x, y, alpha_1, ratio12, ratio13, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22, w_c3, w_t3, e13, e23,
                 center_x=0, center_y=0):
        """

        :param alpha_1:
        :param ratio12: ratio of first to second amplitude
        :param ratio13: ratio of first to third amplidute
        :param w_c1:
        :param w_t1:
        :param e11:
        :param e21:
        :param w_c2:
        :param w_t2:
        :param e12:
        :param e22:
        :param center_x:
        :param center_y:
        :return:
        """
        amp1 = alpha_1 / (1. + 1. / ratio12 + 1. / ratio13)
        amp2 = amp1 / ratio12
        amp3 = amp1 / ratio13
        f_xx1, f_yy1, f_xy1 = self.chameleon.hessian(x, y, amp1, w_c1, w_t1, e11, e21, center_x, center_y)
        f_xx2, f_yy2, f_xy2 = self.chameleon.hessian(x, y, amp2, w_c2, w_t2, e12, e22, center_x, center_y)
        f_xx3, f_yy3, f_xy3 = self.chameleon.hessian(x, y, amp3, w_c3, w_t3, e13, e23, center_x, center_y)
        return f_xx1 + f_xx2 + f_xx3, f_yy1 + f_yy2 + f_yy3, f_xy1 + f_xy2 + f_xy3


class DoubleChameleonPointMass(object):
    """
    class of the Chameleon model (See Suyu+2014) an elliptical truncated double isothermal profile

    """
    param_names = ['alpha_1', 'ratio_chameleon', 'ratio_pointmass', 'w_c1', 'w_t1', 'e11', 'e21', 'w_c2', 'w_t2',
                   'e12', 'e22', 'center_x', 'center_y']
    lower_limit_default = {'alpha_1': 0, 'ratio_chameleon': 0, 'ratio_pointmass': 0, 'w_c1': 0, 'w_t1': 0, 'e11': -0.8,
                           'e21': -0.8, 'w_c2': 0, 'w_t2': 0, 'e12': -0.8, 'e22': -0.8,
                           'center_x': -100, 'center_y': -100}
    upper_limit_default = {'alpha_1': 100, 'ratio_chameleon': 100, 'ratio_pointmass': 100, 'w_c1': 100, 'w_t1': 100, 'e11': 0.8, 'e21': 0.8,
                           'w_c2': 100, 'w_t2': 100, 'e12': 0.8, 'e22': 0.8,
                           'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.chameleon = DoubleChameleon()
        self.pointMass = PointMass()

    def function(self, x, y, alpha_1, ratio_pointmass, ratio_chameleon, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22,
                 center_x=0, center_y=0):
        """
        #TODO chose better parameterization for combining point mass and Chameleon profiles
        :param x: ra-coordinate
        :param y: dec-coordinate
        :param alpha_1: deflection angle at 1 (arcseconds) from the center
        :param ratio_pointmass: ratio of point source Einstein radius to combined Chameleon deflection angle at r=1
        :param ratio_chameleon: ratio in deflection angles at r=1 for the two Chameleon profiles
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return:
        """
        f_1 = self.pointMass.function(x, y, alpha_1 / (1. + 1. / ratio_pointmass), center_x, center_y)
        f_2 = self.chameleon.function(x, y, alpha_1 / (1. + ratio_pointmass), ratio_chameleon, w_c1, w_t1, e11, e21,
                                      w_c2, w_t2, e12, e22, center_x, center_y)
        return f_1 + f_2

    def derivatives(self, x, y, alpha_1, ratio_pointmass, ratio_chameleon, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22,
                    center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param alpha_1:
        :param ratio_pointmass: ratio of point source Einstein radius to combined Chameleon deflection angle at r=1
        :param ratio_chameleon: ratio in deflection angles at r=1 for the two Chameleon profiles
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return:
        """
        f_x1, f_y1 = self.pointMass.derivatives(x, y, alpha_1 / (1. + 1. / ratio_pointmass), center_x, center_y)
        f_x2, f_y2 = self.chameleon.derivatives(x, y, alpha_1 / (1. + ratio_pointmass), ratio_chameleon, w_c1, w_t1,
                                                e11, e21, w_c2, w_t2, e12, e22, center_x, center_y)
        return f_x1 + f_x2, f_y1 + f_y2

    def hessian(self, x, y, alpha_1, ratio_pointmass, ratio_chameleon, w_c1, w_t1, e11, e21, w_c2, w_t2, e12, e22,
                center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param alpha_1:
        :param ratio_pointmass: ratio of point source Einstein radius to combined Chameleon deflection angle at r=1
        :param ratio_chameleon: ratio in deflection angles at r=1 for the two Chameleon profiles
        :param w_c1: Suyu+2014 for first profile
        :param w_t1: Suyu+2014 for first profile
        :param e11: ellipticity parameter for first profile
        :param e21: ellipticity parameter for first profile
        :param w_c2: Suyu+2014 for second profile
        :param w_t2: Suyu+2014 for second profile
        :param e12: ellipticity parameter for second profile
        :param e22: ellipticity parameter for second profile
        :param center_x: ra center
        :param center_y: dec center
        :return:
        """
        f_xx1, f_yy1, f_xy1 = self.pointMass.hessian(x, y, alpha_1 / (1. + 1. / ratio_pointmass), center_x, center_y)
        f_xx2, f_yy2, f_xy2 = self.chameleon.hessian(x, y, alpha_1 / (1. + ratio_pointmass), ratio_chameleon, w_c1, w_t1,
                                                     e11, e21, w_c2, w_t2, e12, e22, center_x, center_y)
        return f_xx1 + f_xx2, f_yy1 + f_yy2, f_xy1 + f_xy2

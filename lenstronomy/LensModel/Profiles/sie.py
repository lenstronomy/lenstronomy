


class SIE(object):
    """
    class for singular isothermal ellipsoid (SIS with ellipticity)
    """
    param_names = ['theta_E', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'theta_E': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self, NIE=False):
        self._nie = NIE
        if NIE:
            from lenstronomy.LensModel.Profiles.nie import NIE
            self.profile = NIE()
        else:
            from lenstronomy.LensModel.Profiles.spemd import SPEMD
            self.profile = SPEMD()
        self._s_scale = 0.0000000001
        self._gamma = 2

    def function(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.function(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.function(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)

    def derivatives(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.derivatives(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.derivatives(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)

    def hessian(self, x, y, theta_E, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param theta_E:
        :param q:
        :param phi_G:
        :param center_x:
        :param center_y:
        :return:
        """
        if self._nie:
            return self.profile.hessian(x, y, theta_E, e1, e2, self._s_scale, center_x, center_y)
        else:
            return self.profile.hessian(x, y, theta_E, self._gamma, e1, e2, center_x, center_y)
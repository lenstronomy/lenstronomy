import lenstronomy.Util.param_util as param_util

__all__ = ['PJaffe', 'PJaffe_Ellipse']


class PJaffe(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution)
    """
    param_names = ['amp', 'Ra', 'Rs', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'Ra': 0, 'Rs': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'Ra': 100, 'Rs': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as PJaffe_lens
        self.lens = PJaffe_lens()

    def function(self, x, y, amp, Ra, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Ra, Rs)
        return self.lens.density_2d(x, y, rho0, Ra, Rs, center_x, center_y)

    def light_3d(self, r, amp, Ra, Rs):
        """

        :param y:
        :param amp:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Ra, Rs)
        return self.lens.density(r, rho0, Ra, Rs)


class PJaffe_Ellipse(object):
    """
    calss for elliptical pseudo Jaffe lens light
    """
    param_names = ['amp', 'Ra', 'Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'Ra': 0, 'Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'Ra': 100, 'Rs': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as PJaffe_lens
        self.lens = PJaffe_lens()
        self.spherical = PJaffe()

    def function(self, x, y, amp, Ra, Rs, e1, e2, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        return self.spherical.function(x_, y_, amp, Ra, Rs)

    def light_3d(self, r, amp, Ra, Rs, e1=0, e2=0):
        """

        :param y:
        :param amp:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.spherical.light_3d(r, amp, Ra, Rs)

import lenstronomy.Util.param_util as param_util

__all__ = ['Hernquist', 'HernquistEllipse']


class Hernquist(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution
    """
    def __init__(self):
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Hernquist_lens
        self.lens = Hernquist_lens()
        self.param_names = ['amp', 'Rs', 'center_x', 'center_y']
        self.lower_limit_default = {'amp': 0, 'Rs': 0, 'center_x': -100, 'center_y': -100}
        self.upper_limit_default = {'amp': 100, 'Rs': 100, 'center_x': 100, 'center_y': 100}

    def function(self, x, y, amp, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param amp:
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density_2d(x, y, rho0, Rs, center_x, center_y)

    def light_3d(self, r, amp, Rs):
        """

        :param y:
        :param amp:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density(r, rho0, Rs)


class HernquistEllipse(object):
    """
    class for elliptical pseudo Jaffe lens light (2d projected light/mass distribution
    """
    param_names = ['amp', 'Rs', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'Rs': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'Rs': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Hernquist_lens
        self.lens = Hernquist_lens()
        self.spherical = Hernquist()

    def function(self, x, y, amp, Rs, e1, e2, center_x=0, center_y=0):
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
        x_, y_ = param_util.transform_e1e2_square_average(x, y, e1, e2, center_x, center_y)
        return self.spherical.function(x_, y_, amp, Rs)

    def light_3d(self, r, amp, Rs, e1=0, e2=0):
        """

        :param y:
        :param amp:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density(r, rho0, Rs)

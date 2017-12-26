import numpy as np


class PJaffe(object):
    """
    class for pseudo Jaffe lens light (2d projected light/mass distribution)
    """
    def __init__(self):
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as PJaffe_lens
        self.lens = PJaffe_lens()

    def function(self, x, y, sigma0, Ra, Rs, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param a:
        :param s:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(sigma0, Ra, Rs)
        return self.lens.density_2d(x, y, rho0, Ra, Rs, center_x, center_y)

    def light_3d(self, r, sigma0, Ra, Rs):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        rho0 = self.lens.sigma2rho(sigma0, Ra, Rs)
        return self.lens.density(r, rho0, Ra, Rs)

class PJaffe_Ellipse(object):
    """
    calss for elliptical pseudo Jaffe lens light
    """
    def __init__(self):
        from lenstronomy.LensModel.Profiles.p_jaffe import PJaffe as PJaffe_lens
        self.lens = PJaffe_lens()
        self.spherical = PJaffe()

    def function(self, x, y, sigma0, Ra, Rs, q, phi_G, center_x=0, center_y=0):
        """

        :param x:
        :param y:
        :param sigma0:
        :param Ra:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        x_, y_ = self._coord_transf(x, y, q, phi_G, center_x, center_y)
        return self.spherical.function(x_, y_, sigma0, Ra, Rs)

    def light_3d(self, r, sigma0, Ra, Rs, q=1, phi_G=0):
        """

        :param y:
        :param sigma0:
        :param Rs:
        :param center_x:
        :param center_y:
        :return:
        """
        return self.spherical.light_3d(r, sigma0, Ra, Rs)

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

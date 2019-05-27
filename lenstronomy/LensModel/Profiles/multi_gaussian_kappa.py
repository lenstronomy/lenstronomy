import numpy as np
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import GaussianEllipsePotential


class MultiGaussianKappa(object):
    """

    """
    param_names = ['amp', 'sigma', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian_kappa = GaussianKappa()

    def function(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_ += self.gaussian_kappa.function(x, y, amp=scale_factor*amp[i], sigma=sigma[i],
                                               center_x=center_x, center_y=center_y)
        return f_

    def derivatives(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_x, f_y = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=scale_factor*amp[i], sigma=sigma[i],
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_xx, f_yy, f_xy = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=scale_factor*amp[i],
                                                                 sigma=sigma[i], center_x=center_x,
                                                                 center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def density(self, r, amp, sigma, scale_factor=1):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        d_ = np.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            d_ += self.gaussian_kappa.density(r, scale_factor*amp[i], sigma[i])
        return d_

    def density_2d(self, x, y, amp, sigma, center_x=0, center_y=0, scale_factor=1):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        d_3d = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            d_3d += self.gaussian_kappa.density_2d(x, y, scale_factor*amp[i], sigma[i], center_x, center_y)
        return d_3d

    def mass_3d_lens(self, R, amp, sigma, scale_factor=1):
        """

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        mass_3d = np.zeros_like(R, dtype=float)
        for i in range(len(amp)):
            mass_3d += self.gaussian_kappa.mass_3d_lens(R, scale_factor*amp[i], sigma[i])
        return mass_3d


class MultiGaussianKappaEllipse(object):
    """

    """
    param_names = ['amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y']
    lower_limit_default = {'amp': 0, 'sigma': 0, 'e1': -0.5, 'e2': -0.5, 'center_x': -100, 'center_y': -100}
    upper_limit_default = {'amp': 100, 'sigma': 100, 'e1': 0.5, 'e2': 0.5, 'center_x': 100, 'center_y': 100}

    def __init__(self):
        self.gaussian_kappa = GaussianEllipsePotential()

    def function(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_ = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_ += self.gaussian_kappa.function(x, y, amp=scale_factor*amp[i], sigma=sigma[i], e1=e1, e2=e2,
                                               center_x=center_x, center_y=center_y)
        return f_

    def derivatives(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_x, f_y = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_x_i, f_y_i = self.gaussian_kappa.derivatives(x, y, amp=scale_factor*amp[i], sigma=sigma[i], e1=e1, e2=e2,
                                                           center_x=center_x, center_y=center_y)
            f_x += f_x_i
            f_y += f_y_i
        return f_x, f_y

    def hessian(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """

        :param x:
        :param y:
        :param amp:
        :param sigma:
        :param center_x:
        :param center_y:
        :return:
        """
        f_xx, f_yy, f_xy = np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float), np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_xx_i, f_yy_i, f_xy_i = self.gaussian_kappa.hessian(x, y, amp=scale_factor*amp[i], sigma=sigma[i], e1=e1, e2=e2,
                                                                 center_x=center_x, center_y=center_y)
            f_xx += f_xx_i
            f_yy += f_yy_i
            f_xy += f_xy_i
        return f_xx, f_yy, f_xy

    def density(self, r, amp, sigma, e1, e2, scale_factor=1):
        """

        :param r:
        :param amp:
        :param sigma:
        :return:
        """
        d_ = np.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            d_ += self.gaussian_kappa.density(r, scale_factor*amp[i], sigma[i], e1, e2)
        return d_

    def density_2d(self, x, y, amp, sigma, e1, e2, center_x=0, center_y=0, scale_factor=1):
        """

        :param R:
        :param am:
        :param sigma_x:
        :param sigma_y:
        :return:
        """
        d_3d = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            d_3d += self.gaussian_kappa.density_2d(x, y, scale_factor*amp[i], sigma[i], e1, e2, center_x, center_y)
        return d_3d

    def mass_3d_lens(self, R, amp, sigma, e1, e2, scale_factor=1):
        """

        :param R:
        :param amp:
        :param sigma:
        :return:
        """
        mass_3d = np.zeros_like(R, dtype=float)
        for i in range(len(amp)):
            mass_3d += self.gaussian_kappa.mass_3d_lens(R, scale_factor*amp[i], sigma[i], e1, e2)
        return mass_3d
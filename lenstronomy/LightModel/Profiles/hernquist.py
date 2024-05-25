import lenstronomy.Util.param_util as param_util
import numpy as np

__all__ = ["Hernquist", "HernquistEllipse"]


class Hernquist(object):
    """Class for Hernquist lens light (2d projected light/mass distribution)."""

    def __init__(self):
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Hernquist_lens

        self.lens = Hernquist_lens()
        self.param_names = ["amp", "Rs", "center_x", "center_y"]
        self.lower_limit_default = {
            "amp": 0,
            "Rs": 0,
            "center_x": -100,
            "center_y": -100,
        }
        self.upper_limit_default = {
            "amp": 100,
            "Rs": 100,
            "center_x": 100,
            "center_y": 100,
        }

    def function(self, x, y, amp, Rs, center_x=0, center_y=0):
        """

        :param x: x-position
        :param y: y-position
        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :param center_x: centroid in x-direction
        :param center_y: centroid in y-direction
        :return: surface brightness
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density_2d(x, y, rho0, Rs, center_x, center_y)

    def light_3d(self, r, amp, Rs):
        """

        :param r: 3d radius (in angular units)
        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :return:
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density(r, rho0, Rs)

    @staticmethod
    def total_flux(amp, Rs):
        """

        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :return: total integrated flux of profile
        """
        rhos = amp / Rs
        m_tot = 2 * np.pi * rhos * Rs**3
        return m_tot


class HernquistEllipse(object):
    """Class for elliptical pseudo Jaffe lens light (2d projected light/mass
    distribution."""

    param_names = ["amp", "Rs", "e1", "e2", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "Rs": 0,
        "e1": -0.5,
        "e2": -0.5,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 100,
        "Rs": 100,
        "e1": 0.5,
        "e2": 0.5,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self):
        from lenstronomy.LensModel.Profiles.hernquist import Hernquist as Hernquist_lens

        self.lens = Hernquist_lens()
        self.spherical = Hernquist()

    def function(self, x, y, amp, Rs, e1, e2, center_x=0, center_y=0):
        """

        :param x: x-position
        :param y: y-position
        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :param e1: eccentricity component
        :param e2: eccentricity component
        :param center_x: centroid in x-direction
        :param center_y: centroid in y-direction
        :return: surface brightness
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x, center_y
        )
        return self.spherical.function(x_, y_, amp, Rs)

    def light_3d(self, r, amp, Rs, e1=0, e2=0):
        """

        :param r: 3d radius (in angular units)
        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: flux density in 3d
        """
        rho0 = self.lens.sigma2rho(amp, Rs)
        return self.lens.density(r, rho0, Rs)

    def total_flux(self, amp, Rs, e1=0, e2=0):
        """

        :param amp: surface brightness amplitude
        :param Rs: scale radius: half-light radius = Rs / 0.551
        :param e1: eccentricity component
        :param e2: eccentricity component
        :return: total integrated flux
        """
        return self.spherical.total_flux(amp=amp, Rs=Rs)

from lenstronomy.LightModel.Profiles.gaussian import Gaussian
import numpy as np


class MGESet(object):
    """Class for Multi Gaussian lens light (2d projected light/mass distribution as a
    set of Gaussians in logarithmic spacing.

    profile name in LightModel module: 'MGE_SET
    """

    param_names = ["amp", "sigma_min", "sigma_width", "center_x", "center_y"]
    lower_limit_default = {
        "amp": 0,
        "sigma_min": 0.0001,
        "sigma_width": 0.001,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 1000,
        "sigma_min": 50,
        "sigma_width": 1000,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, n_comp):
        """

        :param n_comp: number of Gaussian component
        :type n_comp: int > 1
        """
        self._ncomp = int(n_comp)
        assert self._ncomp > 1
        self.gaussian = Gaussian()

    @property
    def num_linear(self):
        """Number of linear parameters.

        :return:
        """
        return self._ncomp

    def _sigma(self, sigma_min, sigma_width):
        """Logarithmically scaled sigmas.

        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :return: array of sigmas for Multi-Gaussian components
        """
        sigmas = np.logspace(
            start=np.log10(sigma_min),
            stop=np.log10(sigma_min + sigma_width),
            num=self._ncomp,
            endpoint=True,
        )
        return sigmas

    def function(self, x, y, amp, sigma_min, sigma_width, center_x=0, center_y=0):
        """Surface brightness per angular unit.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param center_x: center of profile
        :param center_y: center of profile
        :return: surface brightness at (x, y)
        """
        sigma = self._sigma(sigma_min=sigma_min, sigma_width=sigma_width)
        f_ = np.zeros_like(x, dtype=float)
        for i in range(len(amp)):
            f_ += self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
        return f_

    def total_flux(self, amp, sigma_min, sigma_width, center_x=0, center_y=0):
        """Total integrated flux of profile.

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        sigma = self._sigma(sigma_min=sigma_min, sigma_width=sigma_width)
        flux = 0
        for i in range(len(amp)):
            flux += self.gaussian.total_flux(amp[i], sigma[i], center_x, center_y)
        return flux

    def function_split(self, x, y, amp, sigma_min, sigma_width, center_x=0, center_y=0):
        """Split surface brightness in individual components.

        :param x: coordinate on the sky
        :param y: coordinate on the sky
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param center_x: center of profile
        :param center_y: center of profile
        :return: list of arrays of surface brightness
        """
        sigma = self._sigma(sigma_min=sigma_min, sigma_width=sigma_width)
        f_list = []
        for i in range(len(amp)):
            f_list.append(
                self.gaussian.function(x, y, amp[i], sigma[i], center_x, center_y)
            )
        return f_list

    def light_3d(self, r, amp, sigma_min, sigma_width):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :return: 3D brightness per angular volume element
        """
        sigma = self._sigma(sigma_min=sigma_min, sigma_width=sigma_width)
        f_ = np.zeros_like(r, dtype=float)
        for i in range(len(amp)):
            f_ += self.gaussian.light_3d(r, amp[i], sigma[i])
        return f_

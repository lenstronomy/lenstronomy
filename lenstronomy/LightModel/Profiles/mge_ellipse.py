from lenstronomy.LightModel.Profiles.mge_set import MGESet
from lenstronomy.Util import param_util


class MGEEllipse(object):
    """Multi Gaussian Sets with elliptical axis ratios."""

    param_names = [
        "amp",
        "sigma_min",
        "sigma_width",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]
    lower_limit_default = {
        "amp": 0,
        "sigma_min": 0.0001,
        "sigma_width": 0.001,
        "e1": -0.6,
        "e2": -0.6,
        "center_x": -100,
        "center_y": -100,
    }
    upper_limit_default = {
        "amp": 1000,
        "sigma_min": 50,
        "sigma_width": 1000,
        "e1": 0.6,
        "e2": 0.6,
        "center_x": 100,
        "center_y": 100,
    }

    def __init__(self, n_comp):
        """

        :param n_comp: number of Gaussian component
        :type n_comp: int > 1
        """

        self._mge_set = MGESet(n_comp=n_comp)

    @property
    def num_linear(self):
        """Number of linear parameters.

        :return:
        """
        return self._mge_set.num_linear

    def function(
        self, x, y, amp, sigma_min, sigma_width, e1, e2, center_x=0, center_y=0
    ):
        """

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: MGE center x
        :param center_y: MGE center y
        :return: surface brightness
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x=center_x, center_y=center_y
        )
        return self._mge_set.function(
            x_,
            y_,
            amp,
            sigma_min=sigma_min,
            sigma_width=sigma_width,
            center_x=0,
            center_y=0,
        )

    def function_split(
        self, x, y, amp, sigma_min, sigma_width, e1, e2, center_x=0, center_y=0
    ):
        """Slits surface brightness into individual Gaussian components.

        :param x: x-coordinates
        :param y: y-coordinates
        :param amp: array of amplitudes in pre-defined order of shapelet basis functions
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: MGE center x
        :param center_y: MGE center y
        :return: surface brightness split in different components
        """
        x_, y_ = param_util.transform_e1e2_product_average(
            x, y, e1, e2, center_x=center_x, center_y=center_y
        )
        return self._mge_set.function_split(
            x_,
            y_,
            amp,
            sigma_min=sigma_min,
            sigma_width=sigma_width,
            center_x=0,
            center_y=0,
        )

    def total_flux(self, amp, sigma_min, sigma_width, e1, e2, center_x=0, center_y=0):
        """Total integrated flux of profile.

        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :param center_x: center of profile
        :param center_y: center of profile
        :return: total flux
        """
        return self._mge_set.total_flux(
            amp,
            sigma_min=sigma_min,
            sigma_width=sigma_width,
            center_x=center_x,
            center_y=center_y,
        )

    def light_3d(self, r, amp, sigma_min, sigma_width, e1, e2):
        """3D brightness per angular volume element.

        :param r: 3d distance from center of profile
        :param amp: list of amplitudes of individual Gaussian profiles
        :param sigma_min: minimum Gaussian sigma (sigmas being logarithmically scalled
            between min and max)
        :param sigma_width: sigma_min + sigma_width is maximum Gaussian sigma
        :param e1: eccentricity component 1
        :param e2: eccentricity component 2
        :return: 3D brightness per angular volume element
        """
        return self._mge_set.light_3d(
            r, amp, sigma_min=sigma_min, sigma_width=sigma_width
        )

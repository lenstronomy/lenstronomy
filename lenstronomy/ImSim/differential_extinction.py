from lenstronomy.LightModel.light_model import LightModel
import numpy as np


class DifferentialExtinction(object):
    """
    class to compute an extinction (for a specific band/wavelength). This class uses the functionality available in
    the LightModel module to describe an optical depth tau_ext to compute the extinction on the sky/image.
    """

    def __init__(self, optical_depth_model=[]):
        """

        :param optical_depth_model: list of strings naming the profiles (same convention as LightModel module)
        describing the optical depth of the extinction
        """
        self._tau = LightModel(light_model_list=optical_depth_model)
        if len(optical_depth_model) == 0:
            self._compute_bool = False
        else:
            self._compute_bool = True

    def extinction(self, x, y, kwargs_extinction=None, kwargs_special=None):
        """

        :param x: coordinate in image plane of flux intensity
        :param y: coordinate in image plane of flux intensity
        :param tau_0: normalization factor of the extinction profile
        :param kwargs_extinction: keyword argument list matching the extinction profile
        :return: extinction corrected flux
        """
        if self._compute_bool is False or kwargs_extinction is None:
            return 1
        tau = self._tau.surface_brightness(x, y, kwargs_list=kwargs_extinction)
        return np.exp(-tau)

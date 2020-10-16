from lenstronomy.LightModel.light_model import LightModel
import numpy as np

__all__ = ['DifferentialExtinction']


class DifferentialExtinction(object):
    """
    class to compute an extinction (for a specific band/wavelength). This class uses the functionality available in
    the LightModel module to describe an optical depth tau_ext to compute the extinction on the sky/image.
    """

    def __init__(self, optical_depth_model=[], tau0_index=0):
        """

        :param optical_depth_model: list of strings naming the profiles (same convention as LightModel module)
        describing the optical depth of the extinction
        """
        self._profile = LightModel(light_model_list=optical_depth_model)
        if len(optical_depth_model) == 0:
            self._compute_bool = False
        else:
            self._compute_bool = True
        self._tau0_index = tau0_index

    @property
    def compute_bool(self):
        """
        :return: True when a differential extinction is set, False otherwise 
        """
        return self._compute_bool

    def extinction(self, x, y, kwargs_extinction=None, kwargs_special=None):
        """

        :param x: coordinate in image plane of flux intensity
        :param y: coordinate in image plane of flux intensity
        :param tau_0: normalization factor of the extinction profile
        :param kwargs_extinction: keyword argument list matching the extinction profile
        :param kwargs_special: keyword arguments hosting special parameters, here required 'tau0_list'
        :return: extinction corrected flux
        """
        if self._compute_bool is False or kwargs_extinction is None:
            return 1
        tau = self._profile.surface_brightness(x, y, kwargs_list=kwargs_extinction)
        tau0_list = kwargs_special.get('tau0_list', None)
        if tau0_list is not None:
            tau0 = tau0_list[self._tau0_index]
        else:
            tau0 = 1
        return np.exp(-tau0 * tau)

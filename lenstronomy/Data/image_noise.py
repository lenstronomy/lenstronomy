import numpy as np


class ImageNoise(object):
    """
    class that deals with noise properties of imaging data
    """

    def __init__(self, image_data, exposure_time=None, background_rms=None, noise_map=None, verbose=True):
        """

        :param image_data: numpy array, pixel data values
        :param exposure_time: int or array of size the data; exposure time
        (common for all pixels or individually for each individual pixel)
        :param background_rms: root-mean-square value of Gaussian background noise
        :param noise_map: int or array of size the data; joint noise sqrt(variance) of each individual pixel.
        Overwrites meaning of background_rms and exposure_time.
        """
        if exposure_time is not None:
            # make sure no negative exposure values are present no dividing by zero
            if isinstance(exposure_time, int) or isinstance(exposure_time, float):
                if exposure_time <= 10 ** (-10):
                    exposure_time = 10 ** (-10)
            else:
                exposure_time[exposure_time <= 10 ** (-10)] = 10 ** (-10)
        self._exp_map = exposure_time
        self._background_rms = background_rms
        self._noise_map = noise_map
        if noise_map is not None:
            assert np.shape(noise_map) == np.shape(image_data)
        else:
            if background_rms is not None and exposure_time is not None:
                if background_rms * np.max(exposure_time) < 1 and verbose is True:
                    print("WARNING! sigma_b*f %s < 1 count may introduce unstable error estimates with a Gaussian"
                          " error function for a Poisson distribution with mean < 1." % (
                        background_rms * np.max(exposure_time)))
        self._data = image_data

    @property
    def background_rms(self):
        """

        :return: rms value of background noise
        """
        if self._background_rms is None:
            if self._noise_map is None:
                raise ValueError("rms background value as 'background_rms' not specified!")
            self._background_rms = np.median(self._noise_map)
        return self._background_rms

    @property
    def exposure_map(self):
        """
        Units of data and exposure map should result in:
        number of flux counts = data * exposure_map

        :return: exposure map for each pixel
        """
        if self._exp_map is None:
            if self._noise_map is None:
                raise ValueError("Exposure map has not been specified in Noise() class!")
        return self._exp_map

    @property
    def C_D(self):
        """
        Covariance matrix of all pixel values in 2d numpy array (only diagonal component)
        The covariance matrix is estimated from the data.
        WARNING: For low count statistics, the noise in the data may lead to biased estimates of the covariance matrix.

        :return: covariance matrix of all pixel values in 2d numpy array (only diagonal component).
        """
        if not hasattr(self, '_C_D'):
            if self._noise_map is not None:
                self._C_D = self._noise_map ** 2
            else:
                self._C_D = covariance_matrix(self._data, self.background_rms, self.exposure_map)
        return self._C_D

    def C_D_model(self, model):
        """

        :param model: model (same as data but without noise)
        :return: estimate of the noise per pixel based on the model flux
        """
        if self._noise_map is not None:
            return self._noise_map ** 2
        else:
            return covariance_matrix(model, self._background_rms, self._exp_map)


def covariance_matrix(data, background_rms, exposure_map):
    """
    returns a diagonal matrix for the covariance estimation which describes the error

    Notes:

    - the exposure map must be positive definite. Values that deviate too much from the mean exposure time will be
        given a lower limit to not under-predict the Poisson component of the noise.

    - the data must be positive semi-definite for the Poisson noise estimate.
        Values < 0 (Possible after mean subtraction) will not have a Poisson component in their noise estimate.


    :param data: data array, eg in units of photons/second
    :param background_rms: background noise rms, eg. in units (photons/second)^2
    :param exposure_map: exposure time per pixel, e.g. in units of seconds
    :return: len(d) x len(d) matrix that give the error of background and Poisson components; (photons/second)^2
    """
    d_pos = np.zeros_like(data)
    d_pos[data >= 0] = data[data >= 0]
    sigma = d_pos / exposure_map + background_rms ** 2
    return sigma
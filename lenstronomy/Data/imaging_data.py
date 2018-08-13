import numpy as np

from lenstronomy.Data.coord_transforms import Coordinates


class Data(object):
    """
    class to handle the data, coordinate system and masking, including convolution with various numerical precisions

    The Data() class is initialized with keyword arguments:

    - 'image_data': 2d numpy array of the image data
    - 'transform_pix2angle' 2x2 transformation matrix (linear) to transform a pixel shift into a coordinate shift (x, y) -> (ra, dec)
    - 'ra_at_xy_0' RA coordinate of pixel (0,0)
    - 'dec_at_xy_0' DEC coordinate of pixel (0,0)

    optional keywords for shifts in the coordinate system:
    - 'ra_shift': shifts the coordinate system with respect to 'ra_at_xy_0'
    - 'dec_shift': shifts the coordinate system with respect to 'dec_at_xy_0'

    optional keywords for noise properties:
    - 'background_rms': rms value of the background noise
    - 'exp_time': float, exposure time to compute the Poisson noise contribution
    - 'exposure_map': 2d numpy array, effective exposure time for each pixel. If set, will replace 'exp_time'
    - 'noise_map': Gaussian noise (1-sigma) for each individual pixel.
    If this keyword is set, the other noise properties will be ignored.


    Notes:
    ------
    the likelihood for the data given model P(data|model) is defined in the function below. Please make sure that
    your definitions and units of 'exposure_map', 'background_rms' and 'image_data' are in accordance with the
    likelihood function. In particular, make sure that the Poisson noise contribution is defined in the count rate.


    """
    def __init__(self, kwargs_data):
        """

        :param kwargs_data: keyword arguments as described above

        """

        if not 'image_data' in kwargs_data:
            if not 'numPix' in kwargs_data:
                raise ValueError("keyword 'image_data' must be specified and consist of a 2d numpy array  or at least 'numPix'!")
            else:
                numPix = kwargs_data['numPix']
                data = np.zeros((numPix, numPix))
        else:
            data = kwargs_data['image_data']
        self.nx, self.ny = np.shape(data)
        if self.nx != self.ny:
            raise ValueError("'image_data' with non-equal pixel number in x- and y-axis not yet supported!")

        ra_at_xy_0 = kwargs_data.get('ra_at_xy_0', 0) + kwargs_data.get('ra_shift', 0)
        dec_at_xy_0 = kwargs_data.get('dec_at_xy_0', 0) + kwargs_data.get('dec_shift', 0)
        transform_pix2angle = kwargs_data.get('transform_pix2angle', np.array([[1, 0], [0, 1]]))
        self._coords = Coordinates( transform_pix2angle=transform_pix2angle, ra_at_xy_0=ra_at_xy_0,
                                    dec_at_xy_0=dec_at_xy_0)

        self._x_grid, self._y_grid = self._coords.coordinate_grid(self.nx)
        if 'exposure_map' in kwargs_data:
            exp_map = kwargs_data['exposure_map']
            exp_map[exp_map <= 0] = 10**(-10)
        else:
            exp_time = kwargs_data.get('exp_time', 1)
            exp_map = np.ones_like(data) * exp_time
        self._exp_map = exp_map
        self._data = data
        self._sigma_b = kwargs_data.get('background_rms', None)
        if 'noise_map' in kwargs_data:
            self._noise_map = kwargs_data['noise_map']
            if self._noise_map is not None:
                self._sigma_b = 1
                self._exp_map = np.ones_like(data)
        else:
            self._noise_map = None

    def constructor_kwargs(self):
        """


        :return: kwargs that allow to construct the Data() class
        """
        kwargs_data = {'numPix': self.nx, 'image_data': self.data, 'exposure_map': self._exp_map,
                       'background_rms': self._sigma_b, 'ra_at_xy_0': self._coords._ra_at_xy_0,
                        'dec_at_xy_0': self._coords._dec_at_xy_0, 'transform_pix2angle': self._coords._Mpix2a}
        if hasattr(self, '_noise_map'):
            kwargs_data['noise_map'] = self._noise_map
        return kwargs_data

    def update_data(self, image_data):
        """

        update the data

        :param image_data: 2d numpy array of same size as nx, ny
        :return: None
        """
        nx, ny = np.shape(image_data)
        if not self.nx == nx and not self.ny == ny:
            raise ValueError("shape of new data %s %s must equal old data %s %s!" % (nx, ny, self.nx, self.ny))
        self._data = image_data

    @property
    def data(self):
        """

        :return: 2d numpy array of data
        """
        return self._data

    @property
    def deltaPix(self):
        """

        :return: pixel size (in units of arcsec)
        """
        return self._coords.pixel_size

    @property
    def background_rms(self):
        """

        :return: rms value of background noise
        """
        if self._sigma_b is None:
            if self._noise_map is None:
                raise ValueError("rms background value as 'background_rms' not specified!")
        return self._sigma_b

    @property
    def exposure_map(self):
        """
        Units of data and exposure map should result in:
        number of flux counts = data * exposure_map

        :return: exposure map for each pixel
        """
        if self._exp_map is None:
            if self._noise_map is None:
                raise ValueError("Exposure map has not been specified in Data() class!")
        else:
            return self._exp_map

    @property
    def noise_map(self):
        """
        1-sigma error for each pixel (optional)

        :return:
        """
        return self._noise_map

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
                self._C_D = self._noise_map**2
            else:
                self._C_D = self.covariance_matrix(self.data, self.background_rms, self.exposure_map)
        return self._C_D

    @property
    def numData(self):
        """

        :return: number of pixels in the data
        """
        nx, ny = np.shape(self._x_grid)
        return nx*ny

    @property
    def coordinates(self):
        """

        :return: ra and dec coordinates of the pixels, each in 1d numpy arrays
        """
        return self._x_grid, self._y_grid

    def map_coord2pix(self, ra, dec):
        """
        maps the (ra,dec) coordinates of the system into the pixel coordinate of the image

        :param ra: relative RA coordinate as defined by the coordinate frame
        :param dec: relative DEC coordinate as defined by the coordinate frame
        :return: (x, y) pixel coordinates
        """
        return self._coords.map_coord2pix(ra, dec)

    def map_pix2coord(self, x, y):
        """
        maps the (x,y) pixel coordinates of the image into the system coordinates

        :param x: pixel coordinate (can be 1d numpy array), defined in the center of the pixel
        :param y: pixel coordinate (can be 1d numpy array), defined in the center of the pixel
        :return: relative (RA, DEC) coordinates of the system
        """
        return self._coords.map_pix2coord(x, y)

    def covariance_matrix(self, data, background_rms=1, exposure_map=1, noise_map=None, verbose=False):
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
        if noise_map is not None:
            return noise_map**2
        if isinstance(exposure_map, int) or isinstance(exposure_map, float):
            if exposure_map <= 0:
                exposure_map = 1
        else:
            mean_exp_time = np.mean(exposure_map)
            exposure_map[exposure_map < mean_exp_time / 10] = mean_exp_time / 10
        if verbose:
            if background_rms * np.max(exposure_map) < 1:
                print("WARNING! sigma_b*f %s < 1 count may introduce unstable error estimates" % (background_rms * np.max(exposure_map)))
        d_pos = np.zeros_like(data)
        #threshold = 1.5*sigma_b
        d_pos[data >= 0] = data[data >= 0]
        #d_pos[d < threshold] = 0
        sigma = d_pos / exposure_map + background_rms ** 2
        return sigma

    def log_likelihood(self, model, mask, error_map=0):
        """

        computes the likelihood of the data given the model p(data|model)
        The Gaussian errors are estimated with the covariance matrix, based on the model image. The errors include the
        background rms value and the exposure time to compute the Poisson noise level (in Gaussian approximation).

        :param model: the model (same dimensions and units as data)
        :param mask: bool (1, 0) values per pixel. If =0, the pixel is ignored in the likelihood
        :param error_map: additional error term (in same units as covariance matrix).
            This can e.g. come from model errors in the PSF estimation.
        :return: the natural logarithm of the likelihood p(data|model)
        """
        C_D = self.covariance_matrix(model, self._sigma_b, self.exposure_map, self.noise_map)
        X2 = (model - self._data)**2 / (C_D + np.abs(error_map)) * mask
        X2 = np.array(X2)
        logL = - np.sum(X2) / 2
        return logL

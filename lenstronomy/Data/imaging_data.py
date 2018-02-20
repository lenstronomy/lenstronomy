import numpy as np

from lenstronomy.Data.coord_transforms import Coordinates


class Data(object):
    """
    class to handle the data, coordinate system and masking, including convolution with various numerical precisions
    """
    def __init__(self, kwargs_data):
        """

        kwargs_data must contain:

        'image_data': 2d numpy array of the image data
        'transform_pix2angle' 2x2 transformation matrix (linear) to transform a pixel shift into a coordinate shift
        (x, y) -> (ra, dec)
        'ra_at_xy_0' RA coordinate of pixel (0,0)
        'dec_at_xy_0' DEC coordinate of pixel (0,0)

        optional keywords for shifts in the coordinate system:
        'ra_shift': shifts the coordinate system with respect to 'ra_at_xy_0'
        'dec_shift': shifts the coordinate system with respect to 'dec_at_xy_0'

        optional keywords for noise properties:
        'sigma_background': rms value of the background noise
        'exp_time: float, exposure time to compute the Poisson noise contribution
        'exposure_map': 2d numpy array, effective exposure time for each pixel. If set, will replace 'exp_time'

        optional keywords for masking purposes:
        'mask': 2d numpy array consists of zeros or ones. Pixels with mask[i,j]==1 are evaluated in the likelihood process.
        Pixel with mask[i,j]==0 are ignored
        'idex_mask': ignores any computation on the pixels with idex_mask[i,j]==0. Will not be stored in memory during linear inversions.
        Difference to 'mask': Pixels with mask[i,j]==0 will be ray-traced, evaluated and their flux value being
        convolved to enable an impact on other pixels.


        :param kwargs_data:
        :param subgrid_res:
        :param psf_subgrid:
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
            exp_map = kwargs_data.get('exp_time', None)
        self._exp_map = exp_map
        self._data = data
        self._sigma_b = kwargs_data.get('background_rms', None)

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
            raise ValueError("rms background value as 'background_rms' not specified!")
        return self._sigma_b

    @property
    def exposure_map(self):
        """

        :return:
        """
        if self._exp_map is None:
            raise ValueError("Exposure map has not been specified in Data() class!")
        else:
            return self._exp_map

    @property
    def C_D(self):
        """

        :return: covariance matrix of all pixel values in 2d numpy array
        """
        if not hasattr(self, '_C_D'):
            self._C_D = self.covariance_matrix(self.data, self.background_rms, self.exposure_map)
        return self._C_D

    @property
    def numData(self):
        return len(self._x_grid)

    @property
    def coordinates(self):
        return self._x_grid, self._y_grid

    def map_coord2pix(self, ra, dec):
        """

        :param ra:
        :param dec:
        :return:
        """
        return self._coords.map_coord2pix(ra, dec)

    def map_pix2coord(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return self._coords.map_pix2coord(x, y)

    def covariance_matrix(self, d, sigma_b, f, verbose=False):
        """
        returns a diagonal matrix for the covariance estimation
        :param d: data array
        :param sigma_b: background noise
        :param f: reduced poissonian noise
        :return: len(d) x len(d) matrix
        """
        if isinstance(f, int) or isinstance(f, float):
            if f <= 0:
                f = 1
        else:
            mean_exp_time = np.mean(f)
            f[f < mean_exp_time / 10] = mean_exp_time / 10
        if verbose:
            if sigma_b * np.max(f) < 1:
                print("WARNING! sigma_b*f %s >1 may introduce unstable error estimates" % (sigma_b*np.max(f)))
        d_pos = np.zeros_like(d)
        #threshold = 1.5*sigma_b
        d_pos[d >= 0] = d[d >= 0]
        #d_pos[d < threshold] = 0
        sigma = d_pos/f + sigma_b**2
        return sigma

    def log_likelihood(self, model, mask, error_map=0):
        """
        returns reduced residual map
        :param model:
        :param data:
        :param sigma:
        :param reduce_frac:
        :param mask:
        :param error_map:
        :return:
        """
        C_D = self.covariance_matrix(model, self._sigma_b, self.exposure_map)
        X2 = (model - self._data)**2 / (C_D + np.abs(error_map)) * mask
        X2 = np.array(X2)
        logL = - np.sum(X2) / 2
        return logL



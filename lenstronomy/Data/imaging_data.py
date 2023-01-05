import numpy as np

from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.image_noise import ImageNoise

__all__ = ['ImageData']


class ImageData(PixelGrid, ImageNoise):
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

    optional keywords for interferometric quantities:
    - 'log_likelihood_constant': a constant that adds to logL, can be useful for special likelihood computation method
    - 'antenna_primary_beam': primary beam pattern of antennae (now treat each antenna with the same primary beam)
    - 'likelihood_method': can choose 'interferometry_natwt' to compute logL for natwt images, the default one is for gaussian noise

    ** notes **
    the likelihood for the data given model P(data|model) is defined in the function below. Please make sure that
    your definitions and units of 'exposure_map', 'background_rms' and 'image_data' are in accordance with the
    likelihood function. In particular, make sure that the Poisson noise contribution is defined in the count rate.


    """
    def __init__(self, image_data, exposure_time=None, background_rms=None, noise_map=None, gradient_boost_factor=None,
                 ra_at_xy_0=0, dec_at_xy_0=0, transform_pix2angle=None, ra_shift=0, dec_shift=0, log_likelihood_constant = 0,
                 antenna_primary_beam=None, likelihood_method = 'diagonal'):
        """

        :param image_data: 2d numpy array of the image data
        :param exposure_time: int or array of size the data; exposure time
         (common for all pixels or individually for each individual pixel)
        :param background_rms: root-mean-square value of Gaussian background noise in units counts per second
        :param noise_map: int or array of size the data; joint noise sqrt(variance) of each individual pixel.
        :param gradient_boost_factor: None or float, variance terms added in quadrature scaling with
         gradient^2 * gradient_boost_factor
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        :param ra_shift: RA shift of pixel grid
        :param dec_shift: DEC shift of pixel grid
        :param log_likelihood_constant: float, allows user to input a constant that will be added to the log likelihood
        :param antenna_primary_beam: 2d numpy array with the same size of imaga_data;
         more descriptions of the primary beam can be found in the AngularSensitivity class
        :param likelihood_method: string, type of method of log_likelihood computation: options are 'diagonal', 'interferometry_natwt'
         the default option is 'diagonal', which supports gaussian errors with a diagonal noise covariance matrix
        """
        nx, ny = np.shape(image_data)
        if transform_pix2angle is None:
            transform_pix2angle = np.array([[1, 0], [0, 1]])
        PixelGrid.__init__(self, nx, ny, transform_pix2angle, ra_at_xy_0 + ra_shift, dec_at_xy_0 + dec_shift, antenna_primary_beam)
        ImageNoise.__init__(self, image_data, exposure_time=exposure_time, background_rms=background_rms,
                            noise_map=noise_map, gradient_boost_factor=gradient_boost_factor, verbose=False)
        
        self.logL_constant = log_likelihood_constant
        self.logL_method = likelihood_method
        if self.logL_method != 'diagonal' and self.logL_method != 'interferometry_natwt':
            raise ValueError("likelihood_method %s not supported!" % self.logL_method)

    def update_data(self, image_data):
        """

        update the data as well as the error matrix estimated from it when done so using the data

        :param image_data: 2d numpy array of same size as nx, ny
        :return: None
        """
        nx, ny = np.shape(image_data)
        if not self._nx == nx and not self._ny == ny:
            raise ValueError("shape of new data %s %s must equal old data %s %s!" % (nx, ny, self._nx, self._ny))
        self._data = image_data
        if hasattr(self, '_C_D') and self._noise_map is None:
            del self._C_D

    @property
    def data(self):
        """

        :return: 2d numpy array of data
        """
        return self._data

    def log_likelihood(self, model, mask, additional_error_map=0):
        """

        computes the likelihood of the data given the model p(data|model)
        The Gaussian errors are estimated with the covariance matrix, based on the model image. The errors include the
        background rms value and the exposure time to compute the Poisson noise level (in Gaussian approximation).

        :param model: the model (same dimensions and units as data)
        :param mask: bool (1, 0) values per pixel. If =0, the pixel is ignored in the likelihood
        :param additional_error_map: additional error term (in same units as covariance matrix).
            This can e.g. come from model errors in the PSF estimation.
        :return: the natural logarithm of the likelihood p(data|model)
        """
        
        if self.logL_method == 'diagonal':
            C_D = self.C_D_model(model)
            X2 = (model - self._data) ** 2 / (C_D + np.abs(additional_error_map)) * mask
            X2 = np.array(X2)
            logL = - np.sum(X2) / 2
            
        elif self.logL_method == 'interferometry_natwt':
            """
            In this case, the model should be in the form [array1, array2], 
            where array1 and array2 are unconvolved and convolved model images respectively.
            They are both 2d array with the same shape of the data.
            """
            xd = np.sum(model[0] * self.data)
            xAx = np.sum(model[0] * model[1])
            logL = - (xAx - 2 * xd) / (2 * self._background_rms ** 2)
            
        return logL + self.logL_constant
    
    def likelihood_method(self):
        """
        
        passes the likelihood_method to the ImageModel and will be used to identify the method of 
        likelihood computation in ImageLinearFit.
        :return: string, likelihood method
        """
        return self.logL_method

import numpy as np

from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.image_noise import ImageNoise

__all__ = ['ImageData']


class ImageData(PixelGrid, ImageNoise):
    """
    class to handle the data, coordinate system and masking, including convolution with various numerical precisions

    The Data() class is initialized with keyword arguments:

    - 'image_data': 2d numpy array of the image data
    - 'transform_pix2angle' 2x2 transformation matrix (linear) to transform a pixel shift into a coordinate shift
     (x, y) -> (ra, dec)
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
    - 'likelihood_method': need to be specified to 'interferometry_natwt' if one needs to use the interferometric likelihood function.
    The default of 'likelihood_method' is 'diagonal', which is used for non-correlated noises (usually for the CCD images.)
    - 'log_likelihood_constant': a constant that adds to logL.
    - 'antenna_primary_beam': primary beam pattern of antennae (now treat each antenna dish with the same primary beam).

    ** notes **
    the likelihood for the data given model P(data|model) is defined in the function below. Please make sure that
    your definitions and units of 'exposure_map', 'background_rms' and 'image_data' are in accordance with the
    likelihood function. In particular, make sure that the Poisson noise contribution is defined in the count rate.


    """
    def __init__(self, image_data, exposure_time=None, background_rms=None, noise_map=None, gradient_boost_factor=None,
                 ra_at_xy_0=0, dec_at_xy_0=0, transform_pix2angle=None, ra_shift=0, dec_shift=0, phi_rot=0,
                 log_likelihood_constant=0, antenna_primary_beam=None, likelihood_method='diagonal', flux_scaling=1):
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
        :param log_likelihood_constant: float, allows user to input a constant that will be added to the log likelihood. Note that, as for now, this variable is ONLY used for interferometric mode.
        :param antenna_primary_beam: 2d numpy array with the same size of imaga_data;
        :param phi_rot: rotation angle in regard to pixel coordinate transform_pix2angle
        :param antenna_primary_beam: 2d numpy array with the same size of image_data;
         more descriptions of the primary beam can be found in the AngularSensitivity class
        :param likelihood_method: string, type of method of log_likelihood computation: options are 'diagonal', 'interferometry_natwt'.
         The default option 'diagonal' uses a diagonal covariance matrix, which is the case for CCD images.
         The 'interferometry_natwt' option uses our special interferometric likelihood function based on natural weighting images.
        :param flux_scaling: scales the model amplitudes to match the imaging data units. This can be used, for example,
         when modeling multiple exposures that have different magnitude zero points (or flux normalizations) but demand
         the same model normalization
        """
        nx, ny = np.shape(image_data)
        if transform_pix2angle is None:
            transform_pix2angle = np.array([[1, 0], [0, 1]])
        cos_phi, sin_phi = np.cos(phi_rot), np.sin(phi_rot)
        rot_matrix = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
        transform_pix2angle_rot = np.dot(transform_pix2angle, rot_matrix)
        PixelGrid.__init__(self, nx, ny, transform_pix2angle_rot, ra_at_xy_0 + ra_shift, dec_at_xy_0 + dec_shift,
                           antenna_primary_beam)
        ImageNoise.__init__(self, image_data, exposure_time=exposure_time, background_rms=background_rms,
                            noise_map=noise_map, gradient_boost_factor=gradient_boost_factor, verbose=False,
                            flux_scaling=flux_scaling)

        self._logL_constant = log_likelihood_constant
        self._logL_method = likelihood_method
        if self._logL_method != 'diagonal' and self._logL_method != 'interferometry_natwt':
            raise ValueError("likelihood_method %s not supported! likelihood_method can only be 'diagonal' or 'interferometry_natwt'!" % self._logL_method)

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
        # if the likelihood method is assigned to be 'interferometry_natwt', it will return logL computed using the interfermetric likelihood function
        if self._logL_method == 'interferometry_natwt':
            return self.log_likelihood_interferometry(model)

        c_d = self.C_D_model(model)
        chi2 = (model - self._data) ** 2 / (c_d + np.abs(additional_error_map)) * mask
        chi2 = np.array(chi2)
        log_likelihood = - np.sum(chi2) / 2
        return log_likelihood

    def log_likelihood_interferometry(self, model):
        """
        log_likelihood function for natural weighting interferometric images,
        based on (placeholder for Nan Zhang's paper).

        For the interferometry case, the model should be in the form [array1, array2],
        where array1 and array2 are unconvolved and convolved model images respectively.
        They are both 2d array with the same shape of the data.

        The chi^2 of interferometry is computed by
        .. math::
            \\chi^2 =  (d-Ax)^TC^{-1}(d-Ax) = \\frac{1}{\\sigma^2}(d^TA^{-1}d - 2x^Td + x^TAx)
        where :math:`d` and :math:`x` are the data vector and the unconvolved model image vector respectively.
        :math:`A` is the convolution operation matrix, where we normalize the PSF by setting its central pixel to 1.
        :math:`C` is the noise covariance matrix, its diagonal entries are rms^2 of noises, :math:`\\sigma^2`.
        For natural weighting interferometric images, we used the relation
        (see Section 3.2 of https://doi.org/10.1093/mnras/staa2740 for the relation of natural weighting covariance matrix and PSF convolution)
        .. math::
            C = \\sigma^2 A
        to simplify the likelihood function above.
        """

        xd = np.sum(model[0] * self._data)
        xAx = np.sum(model[0] * model[1])
        logL = - (xAx - 2 * xd) / (2 * self._background_rms ** 2) + self._logL_constant
        return logL

    def likelihood_method(self):
        """

        pass the likelihood_method to the ImageModel and will be used to identify the method of
        likelihood computation in ImageLinearFit.
        :return: string, likelihood method
        """
        return self._logL_method

import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution

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


    Notes:
    ------
    the likelihood for the data given model P(data|model) is defined in the function below. Please make sure that
    your definitions and units of 'exposure_map', 'background_rms' and 'image_data' are in accordance with the
    likelihood function. In particular, make sure that the Poisson noise contribution is defined in the count rate.


    """
    def __init__(self, image_data, exposure_time=None, background_rms=None, noise_map=None, gradient_boost_factor=None,
                 ra_at_xy_0=0, dec_at_xy_0=0, transform_pix2angle=None, ra_shift=0, dec_shift=0,
                 primary_beam=None,use_linear_solver=True, likelihood_method = 'diagonal',
                 eigen_vector_set=None,eigen_value_set=None,num_of_modes=None,data_mask=None,
                 d_minv_d = 0, convolution_core = None):
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
        """
        
        """
        primary_beam: 2d np array, the primary beam. It should be in the same size of image_data
        use_linear_solver: bool variable, if True, then the sampler will use the linear solver to find 'amp's

        likelihood_method: should be one of 'diagonal', 'eigen' or 'natwt_special'
                    'diagonal' calculates the diagonal cov matrix likelihood privided by original lenstronomy
                    'eigen' calculates the non-diagonal cov matrix likelihood using eigen modes and eigen values
                    'natwt_special' calculates the natwt PSF convolved image using the speicial method
                    
        The followings are parameters for 'eigen' method:
        eigen_vector_set: 2d np array, eg. eigen_vector_set[0] is the 0th eigen vector
        eigen_value_set: 
            1d np array with entries in a descending sequence,eg. eigen_value_set[0] is the eigenvalue corresponds to eigen_vector_set[0]
            the eigen values are square of rms.
        num_of_modes: a number, the number of modes contributing to the likelihood
        data_mask: a boolean array tells a subset of pixels to do the fitting (the corresponding eigen modes are the constructed from the cov matrix of the same pixel subset)
        
        The followings are parameters for 'natwt_special' method:
        d_minv_d: a constant, (d_minv_d/rms^2) has a constant contribution to the X^2
        convolution_core: the convolution core, i.e. a cut of psf which should be twice larger than the data image.
                        the central pixel should be the brightest and un-renormalized
        """
        
        nx, ny = np.shape(image_data)
        if transform_pix2angle is None:
            transform_pix2angle = np.array([[1, 0], [0, 1]])
        PixelGrid.__init__(self, nx, ny, transform_pix2angle, ra_at_xy_0 + ra_shift, dec_at_xy_0 + dec_shift,primary_beam=None,use_linear_solver=True)
        ImageNoise.__init__(self, image_data, exposure_time=exposure_time, background_rms=background_rms,
                            noise_map=noise_map, gradient_boost_factor=gradient_boost_factor, verbose=False,
                            likelihood_method = 'diagonal',
                            eigen_vector_set=None,eigen_value_set=None,num_of_modes=None,data_mask=None,
                            d_minv_d = 0, convolution_core = None)
        dim=nx*ny
        
        self._bkg_variance = background_rms**2
        
        self._use_linear_solver=use_linear_solver
        self._pb=primary_beam
        if primary_beam is not None:
            pbx,pby=np.shape(primary_beam)
            if (pbx,pby) != (nx,ny):
                raise ValueError("The input primary beam should be in the same size of the data!")
        
        self._likelihood_method = likelihood_method
        self._data_mask=data_mask
        self._eigen_vector_set=eigen_vector_set
        self._eigen_value_set=eigen_value_set
        self._d_minv_d = d_minv_d
        self._convolve_core = convolution_core
        
        if self._likelihood_method == 'eigen':
            if eigen_vector_set is None or eigen_value_set is None:
                raise ValueError("For 'eigen' likelihood method, please input eigen value and eigen vector sets.")
            nv,lv=np.shape(eigen_vector_set)
            num_of_mask=np.nansum(data_mask)
            if lv != dim:
                if self._data_mask is not None:
                    if lv != num_of_mask:
                        raise ValueError("The input eigenvectors should have the same shape with the data mask!")
                else:
                    raise ValueError("The input eigenvectors should have the same dimension with the data!")
            if num_of_modes is None:
                self._num_of_modes=nv
            elif num_of_modes > nv:
                self._num_of_modes=nv
            else:
                self._num_of_modes=num_of_modes
                
        elif self._likelihood_method == 'natwt_special':
            self._convolution = PixelKernelConvolution(kernel = convolution_core)
        elif self._likelihood_method != 'diagonal':
            raise ValueError("The likelihood method should be one of 'diagonal', 'eigen' or 'natwt_special'." )
            
                    
    def check_if_use_linear_solver(self):
        return self._use_linear_solver
    
    def give_pb(self):
        return self._pb

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
        
    
    
        if self._likelihood_method == 'diagonal':           
            C_D = self.C_D_model(model)
            X2 = (model - self._data) ** 2 / (C_D + np.abs(additional_error_map)) * mask
            X2 = np.array(X2)
            logL = - np.sum(X2) / 2
            return logL
        
        elif self._likelihood_method == 'natwt_special':
            xd = np.sum(model * self._data)
            convolved_x = self._convolution._static_fft(model, mode='same')
            xMx = np.sum(model * convolved_x)
            X2_times_variance = self._d_minv_d + xMx - 2*xd
            logL = - 0.5 * X2_times_variance/ (self._bkg_variance)
            return logL
        
        elif self._likelihood_method == 'eigen':
            dchi2_times_variance = np.zeros(self._num_of_modes)
            
            if self._data_mask is not None:
                residual = np.array(model - self._data)
                residual = residual[~np.isnan(self._data_mask)]
            else:
                residual = np.array(util.image2array(model - self._data))  
                
            for i in range(self._num_of_modes):
                coefficient = np.sum(residual * self._eigen_vector_set[i])
                dchi2_times_variance[i] = coefficient * coefficient / self._eigen_value_set[i]
            logL = - 0.5 * np.sum(dchi2_times_variance) / (self._bkg_variance)
            return logL
        
            

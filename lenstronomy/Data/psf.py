import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as util


class PSF(object):
    """
    Point Spread Function convolution
    """

    def __init__(self, psf_type='NONE', fwhm=None, truncation=5, pixel_size=None, kernel_point_source=None,
                 psf_error_map=None, point_source_supersampling_factor=1):
        """

        :param psf_type: string, type of PSF: options are 'NONE', 'PIXEL', 'GAUSSIAN'
        :param fwhm: float, full width at half maximum, only required for 'GAUSSIAN' model
        :param truncation: float, Gaussian truncation (in units of sigma), only required for 'GAUSSIAN' model
        :param pixel_size: width of pixel (required for Gaussian model, not required when using in combination with ImageModel modules)
        :param kernel_point_source: 2d numpy array, odd length, centered PSF of a point source
        :param psf_error_map: uncertainty in the PSF model. Same shape as point source kernel.
        This error will be added to the pixel error around the position of point sources as follows:
        sigma^2_i += 'psf_error_map'_j * (point_source_flux_i)**2
        :param point_source_supersampling_factor: int, supersampling factor of kernel_point_source
        """
        self.psf_type = psf_type
        if self.psf_type == 'GAUSSIAN':
            if fwhm is None:
                raise ValueError('fwhm must be set for GAUSSIAN psf type!')
            self._fwhm = fwhm
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self._truncation = truncation
            self._pixel_size = pixel_size
            self._point_source_supersampling_factor = 0
        elif self.psf_type == 'PIXEL':
            if kernel_point_source is None:
                raise ValueError('kernel_point_source needs to be specified for PIXEL PSF type!')
            if len(kernel_point_source) % 2 == 0:
                raise ValueError('kernel needs to have odd axis number, not ', np.shape(kernel_point_source))
            if point_source_supersampling_factor > 1:
                self._kernel_point_source_supersampled = kernel_point_source
                n_high = len(self._kernel_point_source_supersampled)
                self._point_source_supersampling_factor = point_source_supersampling_factor
                numPix = int(n_high / self._point_source_supersampling_factor)
                if self._point_source_supersampling_factor % 2 == 0:
                    self._kernel_point_source = kernel_util.averaging_even_kernel(self._kernel_point_source_supersampled, self._point_source_supersampling_factor)
                else:
                    kernel_point_source = util.averaging(self._kernel_point_source_supersampled, numGrid=n_high, numPix=numPix)
            else:
                kernel_point_source = kernel_point_source
            self._kernel_point_source = kernel_point_source / np.sum(kernel_point_source)

        elif self.psf_type == 'NONE':
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
        else:
            raise ValueError("psf_type %s not supported!" % self.psf_type)
        if psf_error_map is not None:
            self._psf_error_map = psf_error_map
            if self.psf_type == 'PIXEL':
                if len(self._psf_error_map) != len(self._kernel_point_source):
                    raise ValueError('psf_error_map must have same size as kernel_point_source!')

    @property
    def kernel_point_source(self):
        if not hasattr(self, '_kernel_point_source'):
            if self.psf_type == 'GAUSSIAN':
                kernel_numPix = self._truncation * self._fwhm / self._pixel_size
                if kernel_numPix % 2 == 0:
                    kernel_numPix += 1
                self._kernel_point_source = kernel_util.kernel_gaussian(kernel_numPix, self._pixel_size, self._fwhm)
            else:
                raise ValueError("kernel_point_source could not be created. Please follow the guidelines of the PSF class!")
        return self._kernel_point_source

    @property
    def kernel_pixel(self):
        """
        returns the convolution kernel for a uniform surface brightness on a pixel size

        :return:
        """
        if not hasattr(self, '_kernel_pixel'):
            self._kernel_pixel = kernel_util.pixel_kernel(self.kernel_point_source, subgrid_res=1)
        return self._kernel_pixel

    def kernel_point_source_supersampled(self, supersampling_factor):
        """

        :return:
        """
        if hasattr(self, '_kernel_point_source_supersampled') and self._point_source_supersampling_factor == supersampling_factor:
            pass
        else:
            if self.psf_type == 'GAUSSIAN':
                kernel_numPix = self._truncation / self._pixel_size * supersampling_factor
                kernel_numPix = int(round(kernel_numPix))
                if kernel_numPix % 2 == 0:
                    kernel_numPix += 1
                self._kernel_point_source_supersampled = kernel_util.kernel_gaussian(kernel_numPix, self._pixel_size / supersampling_factor, self._fwhm)
            elif self.psf_type == 'PIXEL':
                kernel = kernel_util.subgrid_kernel(self.kernel_point_source, supersampling_factor, odd=True, num_iter=5)
                n = len(self.kernel_point_source)
                n_new = n * supersampling_factor
                if n_new % 2 == 0:
                    n_new -= 1
                if hasattr(self, '_kernel_point_source_subsampled'):
                    print("Warning: subsampled point source kernel overwritten due to different subsampling size requested.")
                self._kernel_point_source_supersampled = kernel_util.cut_psf(kernel, psf_size=n_new)
                self._point_source_supersampling_factor = supersampling_factor
        return self._kernel_point_source_supersampled

    def set_pixel_size(self, deltaPix):
        """
        update pixel size

        :param deltaPix:
        :return:
        """
        self._pixel_size = deltaPix
        if self.psf_type == 'GAUSSIAN':
            try:
                del self._kernel_point_source
            except:
                pass

    @property
    def psf_error_map(self):
        if not hasattr(self, '_psf_error_map'):
            self._psf_error_map = np.zeros_like(self.kernel_point_source)
        return self._psf_error_map

    @property
    def fwhm(self):
        """

        :return: full width at half maximum of kernel (in units of pixel)
        """
        if self.psf_type == 'GAUSSIAN':
            return self._fwhm / self._pixel_size
        else:
            return kernel_util.fwhm_kernel(self.kernel_point_source)

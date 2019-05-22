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

    def psf_convolution(self, grid, grid_scale, psf_subgrid=False, subgrid_res=1):
        """
        convolves a given pixel grid with a PSF
        """
        psf_type = self.psf_type
        if psf_type == 'NONE':
            return grid
        elif psf_type == 'GAUSSIAN':
            sigma = self._sigma_gaussian/grid_scale
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=self._truncation)
            return img_conv
        elif psf_type == 'PIXEL':
            if psf_subgrid:
                kernel = self.kernel_point_source_supersampled(subgrid_res)
            else:
                kernel = self._kernel_point_source
            img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)

    def psf_convolution_new(self, unconvolved_image, subgrid_res=1, subsampling_size=5, psf_subgrid=True,
                            subgrid_conv_type='grid', conv_type='fft'):
        """

        :param unconvolved_image: 2d image with subsampled pixels with subgrid_res
        :param subgrid_res: subsampling resolution
        :param subsampling_size: size of the subsampling convolution in units of image pixels
        :param psf_subgrid: bool, if False, the convolution is performed on the pixel size of the data
        :param subgrid_conv_type: 'grid' or 'fft', using either scipy.convolve2d or scipy.signal.fftconvolve for
         convolution of subsampled kernel
        :param conv_type: 'grid' or 'fft', using either scipy.convolve2d or scipy.signal.fftconvolve for
         convolution of regular kernel
        :return: convolved 2d image in units of the pixels
        """
        unconvolved_image_resized = image_util.re_size(unconvolved_image, subgrid_res)
        if self.psf_type == 'NONE':
            image_conv_resized = unconvolved_image_resized
        elif self.psf_type == 'GAUSSIAN':
            if psf_subgrid is True:
                grid_scale = self._pixel_size / float(subgrid_res)
                sigma = self._sigma_gaussian/grid_scale
                image_conv = ndimage.filters.gaussian_filter(unconvolved_image, sigma, mode='nearest', truncate=self._truncation)
                image_conv_resized = image_util.re_size(image_conv, subgrid_res)
            else:
                sigma = self._sigma_gaussian / self._pixel_size
                image_conv_resized = ndimage.filters.gaussian_filter(unconvolved_image_resized, sigma, mode='nearest',
                                                             truncate=self._truncation)

        elif self.psf_type == 'PIXEL':
            kernel = self.kernel_point_source
            if subgrid_res > 1 and psf_subgrid is True:
                kernel_subgrid = self.kernel_point_source_supersampled(subgrid_res)
                kernel, kernel_subgrid = kernel_util.split_kernel(kernel, kernel_subgrid, subsampling_size, subgrid_res)
                if subgrid_conv_type == 'fft':
                    image_conv_subgrid = signal.fftconvolve(unconvolved_image, kernel_subgrid, mode='same')
                elif subgrid_conv_type == 'grid':
                    image_conv_subgrid = signal.convolve2d(unconvolved_image, kernel_subgrid, mode='same')
                else:
                    raise ValueError('subgrid_conv_type %s not valid!' % subgrid_conv_type)
                image_conv_resized_1 = image_util.re_size(image_conv_subgrid, subgrid_res)
                if conv_type == 'fft':
                    image_conv_resized_2 = signal.fftconvolve(unconvolved_image_resized, kernel, mode='same')
                elif conv_type == 'grid':
                    image_conv_resized_2 = signal.convolve2d(unconvolved_image_resized, kernel, mode='same')
                else:
                    raise ValueError('conv_type %s not valid!' % conv_type)
                image_conv_resized = image_conv_resized_1 + image_conv_resized_2
            else:
                image_conv_resized = signal.fftconvolve(unconvolved_image_resized, kernel, mode='same')
        else:
            raise ValueError('PSF type %s not valid!' % self.psf_type)
        return image_conv_resized

    @property
    def fwhm(self):
        """

        :return: full width at half maximum of kernel (in units of pixel)
        """
        if self.psf_type == 'GAUSSIAN':
            return self._fwhm / self._pixel_size
        else:
            return kernel_util.fwhm_kernel(self.kernel_point_source)

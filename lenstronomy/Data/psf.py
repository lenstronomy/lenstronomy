import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as util


class PSF(object):
    """
    Point Spread Function convolution

    supported psy_types:

    - 'GAUSSIAN': Gaussian convolution kernel
            kwargs:
                'fwhm': full width at half maximum of Gaussian kernel (arcsec)
            required for point sources:
                'pixel_size': width of kernel in units of pixels
            optional:
                'truncation': truncation of the Gaussian convolution, default: 5*fwhm
                the kernel size will be set to cover everything within the truncation. In units of angle.


    - 'PIXEL': pixelized convolution kernel with odd pixel numbers
            kwargs:
                'kernel_point_source': the pixelized kernel from a point source in its center

            optional:
                'kernel_point_source_subsampled': subsampled point source PSF, this keyword replaces
                    the 'kernel_point_source' with the lower resolution of the subsampled PSF model.

                'point_source_subsampling_factor': int() factor of higher resolution subsampling psf for point source

                'psf_error_map': uncertainty in the PSF model. Same shape as 'kernel_point_source'.
                    This error will be added to the pixel error around the position of point sources as follows:
                     \sigma^2_i += 'psf_error_map'_j * (point_source_flux_i)**2

                'kernel_pixel': PSF of an extended pixel, can be different size (odd axis number) for faster numerical
                    convolution. If not specified, it will take the full point source kernel and numerically convolve it
                    over the size of a pixel

                'kernel_pixel_subsampled': subsampled PSF kernel, this keyword replaces
                    the 'kernel_pixel' with the lower resolution of the subsampled PSF model.

                'pixel_subsampling_factor': int() factor of higher resolution subsampling psf for the convolution kernel

    - 'NONE': default option, results in no convolution, point sources will not be displayed
    """

    def __init__(self, kwargs_psf):
        self.psf_type = kwargs_psf.get('psf_type', 'NONE')
        if self.psf_type == 'GAUSSIAN':
            self._fwhm = kwargs_psf['fwhm']
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self._truncation = kwargs_psf.get('truncation', 5 * self._fwhm)
            if 'pixel_size' in kwargs_psf:
                self._pixel_size = kwargs_psf['pixel_size']
        elif self.psf_type == 'PIXEL':
            if 'kernel_point_source_subsampled' in kwargs_psf:
                self._kernel_point_source_subsampled = kwargs_psf['kernel_point_source_subsampled']
                n_high = len(self._kernel_point_source_subsampled)
                self._point_source_subsampling_factor = kwargs_psf['point_source_subsampling_factor']
                numPix = int(n_high / self._point_source_subsampling_factor)
                self._kernel_point_source = util.averaging(self._kernel_point_source_subsampled, numGrid=n_high, numPix=numPix)
            else:
                self._kernel_point_source = kwargs_psf['kernel_point_source']
            if 'kernel_pixel_subsampled' in kwargs_psf:
                self._kernel_pixel_subsampled = kwargs_psf['kernel_pixel_subsampled']
                n_high = len(self._kernel_point_source_subsampled)
                self._pixel_subsampling_factor = kwargs_psf['pixel_subsampling_factor']
                numPix = int(n_high / self._pixel_subsampling_factor)
                self._kernel_pixel = util.averaging(self._kernel_pixel_subsampled, numGrid=n_high, numPix=numPix)
            else:
                if 'kernel_pixel' in kwargs_psf:
                    self._kernel_pixel = kwargs_psf['kernel_pixel']
                else:
                    self._kernel_pixel = kernel_util.pixel_kernel(self._kernel_point_source, subgrid_res=1)

        elif self.psf_type == 'NONE':
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
        else:
            raise ValueError("psf_type %s not supported!" % self.psf_type)
        if 'psf_error_map' in kwargs_psf:
            self._psf_error_map = kwargs_psf['psf_error_map']
            if len(self._psf_error_map) != len(self._kernel_point_source):
                raise ValueError('psf_error_map must have same size as kernel_point_source!')

    @property
    def kernel_point_source(self):
        if not hasattr(self, '_kernel_point_source'):
            if self.psf_type == 'GAUSSIAN':
                kernel_numPix = self._truncation / self._pixel_size
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
            raise ValueError("kernel_pixel could not be created. Please follow the guidelines of the PSF class!")
        return self._kernel_pixel

    def subgrid_pixel_kernel(self, subgrid_res):
        """

        :return:
        """
        if hasattr(self, '_kernel_pixel_subsampled'):
            if self._pixel_subsampling_factor == subgrid_res:
                pass
        elif hasattr(self, '_kernel_point_source_subsampled'):
            # if the subsampling scale matches the one from the point source - take it!
            if self._point_source_subsampling_factor == subgrid_res:
                self._kernel_pixel_subsampled = self._kernel_point_source_subsampled
                self._pixel_subsampling_factor = self._point_source_subsampling_factor
        else:
            kernel = kernel_util.subgrid_kernel(self.kernel_point_source, subgrid_res, odd=True, num_iter=5)
            n = len(self.kernel_pixel)
            n_new = n * subgrid_res
            if n_new % 2 == 0:
                n_new -= 1
            self._kernel_pixel_subsampled = kernel_util.cut_psf(kernel, psf_size=n_new)
            self._pixel_subsampling_factor = subgrid_res
        return self._kernel_pixel_subsampled

    def subgrid_point_source_kernel(self, subgrid_res):
        """

        :return:
        """
        if hasattr(self, '_kernel_point_source_subsampled'):
            if self._point_source_subsampling_factor == subgrid_res:
                pass
        else:
            kernel = kernel_util.subgrid_kernel(self.kernel_point_source, subgrid_res, odd=True, num_iter=5)
            n = len(self.kernel_point_source)
            n_new = n * subgrid_res
            if n_new % 2 == 0:
                n_new -= 1
            if hasattr(self, '_kernel_point_source_subsampled'):
                print("Warning: subsampled point source kernel overwritten due to different subsampling size requested.")
            self._kernel_point_source_subsampled = kernel_util.cut_psf(kernel, psf_size=n_new)
            self._point_source_subsampling_factor = subgrid_res
        return self._kernel_point_source_subsampled

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
                kernel = self.subgrid_pixel_kernel(subgrid_res)
            else:
                kernel = self._kernel_pixel
            img_conv1 = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv1
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)

    def psf_convolution_new(self, unconvolved_image, subgrid_res=1, subsampling_size=11, psf_subgrid=True):
        """

        :param unconvolved_image: 2d image with subsampled pixels with subgrid_res
        :param subgrid_res: subsampling resolution
        :param subsampling_size: size of the subsampling convolution in units of image pixels
        :param psf_subgrid: bool, if False, the convolution is performed on the pixel size of the data
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
            kernel = self._kernel_pixel
            if subgrid_res > 1 and psf_subgrid is True:
                kernel_subgrid = self.subgrid_pixel_kernel(subgrid_res)
                kernel, kernel_subgrid = kernel_util.split_kernel(kernel, kernel_subgrid, subsampling_size, subgrid_res)
                image_conv_subgrid = signal.fftconvolve(unconvolved_image, kernel_subgrid, mode='same')
                image_conv_resized_1 = image_util.re_size(image_conv_subgrid, subgrid_res)
                image_conv_resized_2 = signal.fftconvolve(unconvolved_image_resized, kernel, mode='same')
                image_conv_resized = image_conv_resized_1 + image_conv_resized_2
            else:
                image_conv_resized = signal.fftconvolve(unconvolved_image_resized, kernel, mode='same')
        else:
            raise ValueError('PSF type %s not valid!' % self.psf_type)
        return image_conv_resized

    @staticmethod
    def psf_fwhm(kwargs, deltaPix):
        """

        :param kwargs: keyword arguments of the PSF() class
        :param deltaPix: pixel size (in the case of subsampled PSF, this is the subsampled pixel size)
        :return: psf fwhm in units of the deltaPix argument
        """
        psf_type = kwargs.get('psf_type', 'NONE')
        if psf_type == 'NONE':
            fwhm = 0
        elif psf_type == 'GAUSSIAN':
            fwhm = kwargs['fwhm']
        elif psf_type == 'PIXEL':
            if 'kernel_point_source_subsampled' in kwargs:
                kernel = kwargs['kernel_point_source_subsampled']
            else:
                kernel = kwargs['kernel_point_source']
            fwhm = kernel_util.fwhm_kernel(kernel) * deltaPix
        else:
            raise ValueError('PSF type %s not valid!' % psf_type)
        return fwhm
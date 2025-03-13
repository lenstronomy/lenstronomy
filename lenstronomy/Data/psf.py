import copy

import numpy as np
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util
import warnings

__all__ = ["PSF"]


class PSF(object):
    """Point Spread Function class.

    This class describes and manages products used to perform the PSF modeling
    (convolution for extended surface brightness and painting of PSF's for point
    sources).
    """

    def __init__(
        self,
        psf_type="NONE",
        fwhm=None,
        truncation=5,
        pixel_size=None,
        kernel_point_source=None,
        psf_variance_map=None,
        point_source_supersampling_factor=1,
        kernel_point_source_init=None,
        kernel_point_source_normalisation=True,
    ):
        """

        :param psf_type: string, type of PSF: options are 'NONE', 'PIXEL', 'GAUSSIAN'
        :param fwhm: float, full width at half maximum, only required for 'GAUSSIAN' model
        :param truncation: float, Gaussian truncation (in units of sigma), only required for 'GAUSSIAN' model
        :param pixel_size: width of pixel (required for Gaussian model, not required when using in combination with
         ImageModel modules)
        :param kernel_point_source: 2d numpy array, odd length, centered PSF of a point source
         (if not normalized, will be normalized)
        :param psf_variance_map: uncertainty in the PSF model per pixel (size of data, not super-sampled). 2d numpy array.
         Size can be larger or smaller than the pixel-sized PSF model and if so, will be matched.
         This error will be added to the pixel error around the position of point sources as follows:
         sigma^2_i += 'psf_variance_map'_j * <point source amplitude>**2
        :param point_source_supersampling_factor: int, supersampling factor of kernel_point_source.
         This is the input PSF to this class and does not need to be the choice in the modeling
         (thought preferred if modeling choses supersampling)
        :param kernel_point_source_init: memory of an initial point source kernel that gets passed through the psf
         iteration
        :param kernel_point_source_normalisation: boolean, if False, the pixel PSF will not be normalized automatically.
        """
        self.psf_type = psf_type
        self._pixel_size = pixel_size
        self.kernel_point_source_init = kernel_point_source_init

        if self.psf_type == "GAUSSIAN":
            if fwhm is None:
                raise ValueError("fwhm must be set for GAUSSIAN psf type!")
            self._fwhm = fwhm
            self._sigma_gaussian = util.fwhm2sigma(self._fwhm)
            self.truncation = truncation
            self._point_source_supersampling_factor = 0
        elif self.psf_type == "PIXEL":
            if kernel_point_source is None:
                raise ValueError(
                    "kernel_point_source needs to be specified for PIXEL PSF type!"
                )
            if len(kernel_point_source) % 2 == 0:
                raise ValueError(
                    "kernel needs to have odd axis number, not ",
                    np.shape(kernel_point_source),
                )
            # store the initial input PSF and supersampling factor
            self._kernel_point_source_init = kernel_point_source
            self._point_source_supersampling_factor_init = (
                point_source_supersampling_factor
            )
            kernel_point_source_ = copy.deepcopy(kernel_point_source)
            if kernel_point_source_normalisation is True:
                kernel_point_source_ /= np.sum(kernel_point_source)
            if point_source_supersampling_factor > 1:
                self._kernel_point_source_supersampled = kernel_point_source_
                self._point_source_supersampling_factor = (
                    point_source_supersampling_factor
                )
                kernel_point_source_ = kernel_util.degrade_kernel(
                    self._kernel_point_source_supersampled,
                    self._point_source_supersampling_factor,
                )
                if kernel_point_source_normalisation is False:
                    kernel_point_source_ *= np.sum(kernel_point_source) / np.sum(
                        kernel_point_source_
                    )
            # making sure the PSF is positive semi-definite and do the normalisation if kernel_point_source_normalisation is true
            if np.min(kernel_point_source_) < 0:
                warnings.warn(
                    "Input PSF model has at least one negative element, which is unphysical except for a PSF of "
                    "an interferometric array."
                )
            self._kernel_point_source = kernel_point_source_

        elif self.psf_type == "NONE":
            self._kernel_point_source = np.zeros((3, 3))
            self._kernel_point_source[1, 1] = 1
        else:
            raise ValueError("psf_type %s not supported!" % self.psf_type)
        if psf_variance_map is not None:
            n_kernel = len(self.kernel_point_source)
            self._psf_variance_map = kernel_util.match_kernel_size(
                psf_variance_map, n_kernel
            )
            if self.psf_type == "PIXEL" and point_source_supersampling_factor > 1:
                if len(psf_variance_map) == len(self._kernel_point_source_supersampled):
                    Warning(
                        "psf_variance_map has the same size as the super-sampled kernel. Make sure the units in the"
                        "psf_variance_map are on the down-sampled pixel scale."
                    )
            if kernel_point_source_normalisation is True:
                self._psf_variance_map /= np.sum(kernel_point_source) ** 2
            self.psf_variance_map_bool = True
        else:
            self.psf_variance_map_bool = False

        self._kernel_point_source_normalisation = kernel_point_source_normalisation
        if kernel_point_source_normalisation is False and psf_type == "PIXEL":
            self._kernel_norm = np.sum(kernel_point_source)
        else:
            self._kernel_norm = 1

    @property
    def kernel_point_source(self):
        if not hasattr(self, "_kernel_point_source"):
            if self.psf_type == "GAUSSIAN":
                sigma = util.fwhm2sigma(self._fwhm)
                # This num_pix definition is equivalent to that of the scipy ndimage.gaussian_filter
                # num_pix = 2r + 1 where r = round(truncation * sigma) is the radius of the gaussian kernel
                # kernel_num_pix is always an odd integer between 3 and 221
                kernel_radius = max(
                    round(self.truncation * sigma / self._pixel_size), 1
                )
                kernel_num_pix = min(2 * kernel_radius + 1, 221)
                self._kernel_point_source = kernel_util.kernel_gaussian(
                    kernel_num_pix, self._pixel_size, self._fwhm
                )
        return self._kernel_point_source

    @property
    def kernel_pixel(self):
        """Returns the convolution kernel for a uniform surface brightness on a pixel
        size.

        :return: 2d numpy array
        """
        if not hasattr(self, "_kernel_pixel"):
            kernel_pixel = kernel_util.pixel_kernel(
                self.kernel_point_source, subgrid_res=1
            )
            self._kernel_pixel = kernel_pixel * self._kernel_norm / np.sum(kernel_pixel)
        return self._kernel_pixel

    def kernel_point_source_supersampled(self, supersampling_factor, updata_cache=True):
        """Generates (if not already available) a supersampled PSF with odd numbers of
        pixels centered.

        :param supersampling_factor: int >=1, supersampling factor relative to pixel
            resolution
        :param updata_cache: boolean, if True, updates the cached supersampling PSF if
            generated. Attention, this will overwrite a previously used supersampled PSF
            if the resolution is changing.
        :return: super-sampled PSF as 2d numpy array
        """
        if supersampling_factor == 1:
            return self.kernel_point_source
        if (
            hasattr(self, "_kernel_point_source_supersampled")
            and self._point_source_supersampling_factor == supersampling_factor
        ):
            kernel_point_source_supersampled = self._kernel_point_source_supersampled
            return kernel_point_source_supersampled
        if hasattr(self, "_kernel_point_source_init") and hasattr(
            self, "_point_source_supersampling_factor_init"
        ):
            if self._point_source_supersampling_factor_init == supersampling_factor:
                kernel_point_source_supersampled = self._kernel_point_source_init
                return kernel_point_source_supersampled

        if self.psf_type == "GAUSSIAN":
            sigma = util.fwhm2sigma(self._fwhm)
            # This num_pix definition is equivalent to that of the scipy ndimage.gaussian_filter
            # num_pix = 2r + 1 where r = round(truncation * sigma) is the radius of the gaussian kernel
            kernel_radius = max(
                round(
                    self.truncation * sigma / self._pixel_size * supersampling_factor
                ),
                1,
            )
            kernel_num_pix = 2 * kernel_radius + 1
            if kernel_num_pix > 10000:
                raise ValueError(
                    "The pixelized Gaussian kernel has a grid of %s pixels with a truncation at "
                    "%s times the sigma of the Gaussian, exceeding the limit allowed."
                    % (kernel_num_pix, self.truncation)
                )
            kernel_point_source_supersampled = kernel_util.kernel_gaussian(
                kernel_num_pix, self._pixel_size / supersampling_factor, self._fwhm
            )

        elif self.psf_type == "PIXEL":

            kernel = kernel_util.subgrid_kernel(
                self.kernel_point_source, supersampling_factor, odd=True, num_iter=5
            )
            n = len(self.kernel_point_source)
            n_new = n * supersampling_factor
            if n_new % 2 == 0:
                n_new -= 1
            if hasattr(self, "_kernel_point_source_supersampled"):
                warnings.warn(
                    "Super-sampled point source kernel over-written due to different subsampling"
                    " size requested. Previous supersampling factor: %s. New supersampling factor %s"
                    % (self._point_source_supersampling_factor, supersampling_factor),
                    Warning,
                )
            kernel_point_source_supersampled = kernel_util.cut_psf(
                kernel, psf_size=n_new
            )
            kernel_point_source_supersampled *= self._kernel_norm / np.sum(
                kernel_point_source_supersampled
            )

        elif self.psf_type == "NONE":
            kernel_point_source_supersampled = self._kernel_point_source
        else:
            raise ValueError("psf_type %s not valid!" % self.psf_type)
        if updata_cache is True:
            self._kernel_point_source_supersampled = kernel_point_source_supersampled
            self._point_source_supersampling_factor = supersampling_factor
        return kernel_point_source_supersampled

    def set_pixel_size(self, deltaPix):
        """Update pixel size.

        :param deltaPix: pixel size in angular units (arc seconds)
        :return: None
        """
        self._pixel_size = deltaPix
        if self.psf_type == "GAUSSIAN":
            try:
                del self._kernel_point_source
            except:
                pass

    @property
    def psf_variance_map(self):
        """Error variance of the normalized PSF.

        This error will be added to the pixel error around the position of point sources as follows:
        sigma^2_i += 'psf_variance_map'_j * <point source amplitude>**2

        :return: error variance of the normalized PSF. Variance of
        :rtype: 2d numpy array of size of the PSF in pixel size (not supersampled)
        """
        if not hasattr(self, "_psf_variance_map"):
            self._psf_variance_map = np.zeros_like(self.kernel_point_source)
        return self._psf_variance_map

    @property
    def fwhm(self):
        """

        :return: full width at half maximum of kernel (in units of angle)
        """
        if self.psf_type == "GAUSSIAN":
            return self._fwhm
        else:
            return kernel_util.fwhm_kernel(self.kernel_point_source) * self._pixel_size

    @property
    def point_source_supersampling_factor(self):
        """

        :return: supersampling factor of initial PSF (if PIXEL type), otherwise 1
        """
        if hasattr(self, "_point_source_supersampling_factor_init"):
            return self._point_source_supersampling_factor_init
        else:
            return 1

from lenstronomy.ImSim.Numerics.grid import RegularGrid, AdaptiveGrid
from lenstronomy.ImSim.Numerics.convolution import SubgridKernelConvolution, PixelKernelConvolution, MultiGaussianConvolution
from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering
from lenstronomy.Util import util
from lenstronomy.Util import kernel_util
import numpy as np

__all__ = ['Numerics']


class Numerics(PointSourceRendering):
    """
    this classes manages the numerical options and computations of an image.
    The class has two main functions, re_size_convolve() and coordinates_evaluate()
    """
    def __init__(self, pixel_grid, psf, supersampling_factor=1, compute_mode='regular', supersampling_convolution=False,
                 supersampling_kernel_size=5, flux_evaluate_indexes=None, supersampled_indexes=None,
                 compute_indexes=None, point_source_supersampling_factor=1, convolution_kernel_size=None,
                 convolution_type='fft_static', truncation=4):
        """

        :param pixel_grid: PixelGrid() class instance
        :param psf: PSF() class instance
        :param compute_mode: options are: 'regular', 'adaptive'
        :param supersampling_factor: int, factor of higher resolution sub-pixel sampling of surface brightness
        :param supersampling_convolution: bool, if True, performs (part of) the convolution on the super-sampled
        grid/pixels
        :param supersampling_kernel_size: int (odd number), size (in regular pixel units) of the super-sampled
        convolution
        :param flux_evaluate_indexes: boolean 2d array of size of image (or None, then initiated as gird of True's).
        Pixels indicated with True will be used to perform the surface brightness computation (and possible lensing
        ray-shooting). Pixels marked as False will be assigned a flux value of zero (or ignored in the adaptive
        convolution)
        :param supersampled_indexes: 2d boolean array (only used in mode='adaptive') of pixels to be supersampled (in
        surface brightness and if supersampling_convolution=True also in convolution). All other pixels not set to =True
        will not be super-sampled.
        :param compute_indexes: 2d boolean array (only used in compute_mode='adaptive'), marks pixel that the response after
        convolution is computed (all others =0). This can be set to likelihood_mask in the Likelihood module for
        consistency.
        :param point_source_supersampling_factor: super-sampling resolution of the point source placing
        :param convolution_kernel_size: int, odd number, size of convolution kernel. If None, takes size of point_source_kernel
        :param convolution_type: string, 'fft', 'grid', 'fft_static' mode of 2d convolution
        """
        if compute_mode not in ['regular', 'adaptive']:
            raise ValueError('compute_mode specified as %s not valid. Options are "adaptive", "regular"')
        # if no super sampling, turn the supersampling convolution off
        self._psf_type = psf.psf_type
        if not isinstance(supersampling_factor, int):
            raise TypeError('supersampling_factor needs to be an integer! Current type is %s' % type(supersampling_factor))
        if supersampling_factor == 1:
            supersampling_convolution = False
        self._pixel_width = pixel_grid.pixel_width
        nx, ny = pixel_grid.num_pixel_axes
        transform_pix2angle = pixel_grid.transform_pix2angle
        ra_at_xy_0, dec_at_xy_0 = pixel_grid.radec_at_xy_0
        if supersampled_indexes is None:
            supersampled_indexes = np.zeros((nx, ny), dtype=bool)
        if compute_mode == 'adaptive':  # or (compute_mode == 'regular' and supersampling_convolution is False and supersampling_factor > 1):
            self._grid = AdaptiveGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampled_indexes,
                                      supersampling_factor, flux_evaluate_indexes)
        else:
            self._grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_factor,
                                     flux_evaluate_indexes)
        if self._psf_type == 'PIXEL':
            if compute_mode == 'adaptive' and supersampling_convolution is True:
                from lenstronomy.ImSim.Numerics.adaptive_numerics import AdaptiveConvolution
                kernel_super = psf.kernel_point_source_supersampled(supersampling_factor)
                kernel_super = self._supersampling_cut_kernel(kernel_super, convolution_kernel_size, supersampling_factor)
                self._conv = AdaptiveConvolution(kernel_super, supersampling_factor,
                                                 conv_supersample_pixels=supersampled_indexes,
                                                 supersampling_kernel_size=supersampling_kernel_size,
                                                 compute_pixels=compute_indexes, nopython=True, cache=True, parallel=False)

            elif compute_mode == 'regular' and supersampling_convolution is True:
                kernel_super = psf.kernel_point_source_supersampled(supersampling_factor)
                if convolution_kernel_size is not None:
                    kernel_super = psf.kernel_point_source_supersampled(supersampling_factor)
                    kernel_super = self._supersampling_cut_kernel(kernel_super, convolution_kernel_size,
                                                                  supersampling_factor)
                self._conv = SubgridKernelConvolution(kernel_super, supersampling_factor,
                                                      supersampling_kernel_size=supersampling_kernel_size,
                                                      convolution_type=convolution_type)
            else:
                kernel = psf.kernel_point_source
                kernel = self._supersampling_cut_kernel(kernel, convolution_kernel_size,
                                                              supersampling_factor=1)
                self._conv = PixelKernelConvolution(kernel, convolution_type=convolution_type)

        elif self._psf_type == 'GAUSSIAN':
            pixel_scale = pixel_grid.pixel_width
            fwhm = psf.fwhm  # FWHM  in units of angle
            sigma = util.fwhm2sigma(fwhm)
            sigma_list = [sigma]
            fraction_list = [1]
            self._conv = MultiGaussianConvolution(sigma_list, fraction_list, pixel_scale, supersampling_factor,
                                                  supersampling_convolution, truncation=truncation)
        elif self._psf_type == 'NONE':
            self._conv = None
        else:
            raise ValueError('psf_type %s not valid! Chose either NONE, GAUSSIAN or PIXEL.' % self._psf_type)
        super(Numerics, self).__init__(pixel_grid=pixel_grid, supersampling_factor=point_source_supersampling_factor,
                                       psf=psf)
        if supersampling_convolution is True:
            self._high_res_return = True
        else:
            self._high_res_return = False

    def re_size_convolve(self, flux_array, unconvolved=False):
        """

        :param flux_array: 1d array, flux values corresponding to coordinates_evaluate
        :param array_low_res_partial: regular sampled surface brightness, 1d array
        :return: convolved image on regular pixel grid, 2d array
        """
        # add supersampled region to lower resolution on
        image_low_res, image_high_res_partial = self._grid.flux_array2image_low_high(flux_array,
                                                                                     high_res_return=self._high_res_return)
        if unconvolved is True or self._psf_type == 'NONE':
            image_conv = image_low_res
        else:
            # convolve low res grid and high res grid
            image_conv = self._conv.re_size_convolve(image_low_res, image_high_res_partial)
        return image_conv * self._pixel_width ** 2

    @property
    def grid_supersampling_factor(self):
        """

        :return: supersampling factor set for higher resolution sub-pixel sampling of surface brightness
        """
        return self._grid.supersampling_factor

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._grid.coordinates_evaluate

    @staticmethod
    def _supersampling_cut_kernel(kernel_super, convolution_kernel_size, supersampling_factor):
        """

        :param kernel_super: super-sampled kernel
        :param convolution_kernel_size: size of convolution kernel in units of regular pixels (odd)
        :param supersampling_factor: super-sampling factor of convolution kernel
        :return: cut out kernel in super-sampling size
        """
        if convolution_kernel_size is not None:
            size = convolution_kernel_size * supersampling_factor
            if size % 2 == 0:
                size += 1
            kernel_cut = kernel_util.cut_psf(kernel_super, size)
            return kernel_cut
        else:
            return kernel_super

    @property
    def convolution_class(self):
        """

        :return: convolution class (can be SubgridKernelConvolution, PixelKernelConvolution, MultiGaussianConvolution, ...)
        """
        return self._conv

    @property
    def grid_class(self):
        """

        :return: grid class (can be RegularGrid, AdaptiveGrid)
        """
        return self._grid

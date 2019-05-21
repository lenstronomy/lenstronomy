from lenstronomy.ImSim.Numerics.adaptive_numerics import AdaptiveConvolution
from lenstronomy.ImSim.Numerics.grid import RegularGrid, AdaptiveGrid
import numpy as np


class Numerics(object):
    """
    this classes manages the numerical options and computations of an image.
    The class has two main functions, re_size_convolve() and coordinates_evaluate()
    """
    def __init__(self, pixel_grid, psf, supersampling_factor=1, compute_mode='regular', supersampling_convolution=True,
                 supersampling_kernel_size=5, flux_evaluate_indexes=None, supersampled_indexes=None,
                 compute_indexes=None):
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
        surface brightness and if supersampling_convolution=True also in convolution)
        :param compute_indexes: 2d boolean array (only used in mode='adaptive'), marks pixel that the resonse after
        convolution is computed (all others =0). This can be set to likelihood_mask in the Likelihood module for
        consistency.

        """
        # if no super sampling, turn the supersampling convolution off
        if supersampling_factor == 1:
            supersampling_convolution = False
        nx, ny = pixel_grid.num_pixel_axes
        transform_pix2angle = pixel_grid.transform_pix2angle
        ra_at_xy_0, dec_at_xy_0 = pixel_grid.radec_at_xy_0
        if supersampled_indexes is None:
            supersampled_indexes = np.zeros((nx, ny), dtype=bool)
        if compute_mode == 'adaptive' or (compute_mode == 'regular' and supersampling_convolution is False and supersampling_factor > 1):
            self._grid = AdaptiveGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampled_indexes,
                                      supersampling_factor, flux_evaluate_indexes)
        else:
            self._grid = RegularGrid(nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_factor,
                                     flux_evaluate_indexes)
        if compute_mode == 'adaptive' and supersampling_convolution is True and supersampling_factor > 1:
            kernel_super = psf.subgrid_point_source_kernel(supersampling_factor)
            self._conv = AdaptiveConvolution(kernel_super, supersampling_factor,
                                             conv_supersample_pixels=supersampled_indexes,
                                             supersampling_kernel_size=supersampling_kernel_size,
                                             compute_pixels=compute_indexes, nopython=True, cache=True, parallel=False)

        elif compute_mode == 'regular':
            pass
            #self._conv = RegularConvolution(supersampling_factor=supersampling_factor,
            #                                supersampling_kernel_size=supersampling_kernel_size)
        else:
            raise ValueError('compute_mode %s not valid! Chose either regular or adaptive.' % compute_mode)

    def re_size_convolve(self, flux_array):
        """

        :param flux_array: 1d array, flux values corresponding to coordinates_evaluate
        :param array_low_res_partial: regular sampled surface brightness, 1d array
        :return: convolved image on regular pixel grid, 2d array
        """
        # add supersampled region to lower resolution on
        image_low_res, image_high_res_partial = self._grid.flux_array2image_low_high(flux_array)
        # convolve low res grid and high res grid
        image_conv = self._conv.re_size_convolve(image_low_res, image_high_res_partial)
        return image_conv

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._grid.coordinates_evaluate
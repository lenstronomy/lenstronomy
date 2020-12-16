import numpy as np
from lenstronomy.ImSim.Numerics.numerics import Numerics
from lenstronomy.ImSim.Numerics.point_source_rendering import PointSourceRendering
from lenstronomy.Data.pixel_grid import PixelGrid

__all__ = ['NumericsSubFrame']


class NumericsSubFrame(PointSourceRendering):
    """
    This class finds the optimal rectangular sub-frame of a data to be modelled that contains all the
    flux_evaluate_indexes and performs the numerical calculations only in this frame and then patches zeros around it
    to match the full data size.
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
        surface brightness and if supersampling_convolution=True also in convolution)
        :param compute_indexes: 2d boolean array (only used in mode='adaptive'), marks pixel that the resonse after
        convolution is computed (all others =0). This can be set to likelihood_mask in the Likelihood module for
        consistency.
        :param point_source_supersampling_factor: super-sampling resolution of the point source placing
        :param convolution_kernel_size: int, odd number, size of convolution kernel. If None, takes size of
        point_source_kernel

        """
        # if no super sampling, turn the supersampling convolution off

        self._nx, self._ny = pixel_grid.num_pixel_axes
        self._init_sub_frame(flux_evaluate_indexes)
        pixel_grid_sub = self._sub_pixel_grid(pixel_grid)
        self._numerics_subframe = Numerics(pixel_grid=pixel_grid_sub, psf=psf,
                                           supersampling_factor=supersampling_factor, compute_mode=compute_mode,
                                           supersampling_convolution=supersampling_convolution,
                                           supersampling_kernel_size=supersampling_kernel_size,
                                           flux_evaluate_indexes=self._cut_frame(flux_evaluate_indexes),
                                           supersampled_indexes=self._cut_frame(supersampled_indexes),
                                           compute_indexes=self._cut_frame(compute_indexes),
                                           point_source_supersampling_factor=point_source_supersampling_factor,
                                           convolution_kernel_size=convolution_kernel_size,
                                           convolution_type=convolution_type, truncation=truncation)
        super(NumericsSubFrame, self).__init__(pixel_grid=pixel_grid, supersampling_factor=point_source_supersampling_factor,
                                       psf=psf)

    def re_size_convolve(self, flux_array, unconvolved=False):
        """

        :param flux_array: 1d array, flux values corresponding to coordinates_evaluate
        :param array_low_res_partial: regular sampled surface brightness, 1d array
        :return: convolved image on regular pixel grid, 2d array
        """
        # add supersampled region to lower resolution on
        image_sub_frame = self._numerics_subframe.re_size_convolve(flux_array, unconvolved=unconvolved)
        return self._complete_frame(image_sub_frame)

    @property
    def grid_supersampling_factor(self):
        """

        :return: supersampling factor set for higher resolution sub-pixel sampling of surface brightness
        """
        return self._numerics_subframe.grid_supersampling_factor

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        return self._numerics_subframe.coordinates_evaluate

    @property
    def convolution_class(self):
        """

        :return: convolution class (can be SubgridKernelConvolution, PixelKernelConvolution, MultiGaussianConvolution, ...)
        """
        return self._numerics_subframe.convolution_class

    @property
    def grid_class(self):
        """

        :return: grid class (can be RegularGrid, AdaptiveGrid)
        """
        return self._numerics_subframe.grid_class

    def _complete_frame(self, image_sub_frame):
        """
        :param image_sub_frame: 2d numpy array of size of the sub-frame
        :return: 2d numpy array of size of image with added zeros on their edges
        """
        if self._subframe_calc is True:
            image = np.zeros((self._nx, self._ny))
            image[self._x_min_sub:self._x_max_sub + 1, self._y_min_sub:self._y_max_sub + 1] = image_sub_frame
        else:
            image = image_sub_frame
        return image

    def _init_sub_frame(self, flux_evaluate_indexes):
        """
        smaller frame that encolses all the idex_mask
        :param idex_mask:
        :param nx:
        :param ny:
        :return:
        """
        if flux_evaluate_indexes is None:
            self._subframe_calc = False
            self._x_min_sub, self._y_min_sub = 0, 0
            self._x_max_sub, self._y_max_sub = self._nx, self._ny
        else:
            self._subframe_calc = True
            self._x_min_sub = np.min(np.where(flux_evaluate_indexes)[0])
            self._x_max_sub = np.max(np.where(flux_evaluate_indexes)[0])
            self._y_min_sub = np.min(np.where(flux_evaluate_indexes)[1])
            self._y_max_sub = np.max(np.where(flux_evaluate_indexes)[1])

    def _cut_frame(self, image):
        """

        :param image: 2d array of full image size
        :return: 2d array of the sub-frame
        """
        if self._subframe_calc is True and image is not None:
            return image[self._x_min_sub:self._x_max_sub + 1, self._y_min_sub:self._y_max_sub + 1]
        else:
            return image

    def _sub_pixel_grid(self, pixel_grid):
        """
        creates a PixelGrid instance covering the sub-frame area only

        :param pixel_grid: PixelGrid instance of the full image
        :return: PixelGrid instance
        """
        if self._subframe_calc is True:
            transform_pix2angle = pixel_grid.transform_pix2angle
            nx_sub = self._x_max_sub - self._x_min_sub + 1
            ny_sub = self._y_max_sub - self._y_min_sub + 1
            ra_at_xy_0_sub, dec_at_xy_0_sub = pixel_grid.map_pix2coord(self._x_min_sub, self._y_min_sub)
            pixel_grid_sub = PixelGrid(nx=nx_sub, ny=ny_sub, transform_pix2angle=transform_pix2angle,
                                       ra_at_xy_0=ra_at_xy_0_sub, dec_at_xy_0=dec_at_xy_0_sub)
        else:
            pixel_grid_sub = pixel_grid
        return pixel_grid_sub

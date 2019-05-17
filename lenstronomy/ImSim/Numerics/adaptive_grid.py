import numpy as np
import lenstronomy.Util.util as util
from lenstronomy.Data.coord_transforms import Coordinates1D


class AdaptiveGrid(Coordinates1D):
    """
    manages a super-sampled grid on the partial image
    """
    def __init__(self, nx, ny, transform_pix2angle, ra_at_xy_0, dec_at_xy_0, supersampling_indexes,
                 supersampling_factor, compute_indexes=None):
        """

        :param nx: number of pixels in x-axis
        :param ny: number of pixels in y-axis
        :param transform_pix2angle: 2x2 matrix, mapping of pixel to coordinate
        :param ra_at_xy_0: ra coordinate at pixel (0,0)
        :param dec_at_xy_0: dec coordinate at pixel (0,0)
        :param supersampling_indexes: bool array of shape nx x ny, corresponding to pixels being super_sampled
        :param compute_indexes: bool array of shape nx x ny, corresponding to pixels being evaluated
        (for both low and high res). Default is None, replaced by setting all pixels to being evaluated.
        """
        super(AdaptiveGrid, self).__init__(transform_pix2angle, ra_at_xy_0, dec_at_xy_0)
        self._nx = nx
        self._ny = ny
        self._x_grid, self._y_grid = self.coordinate_grid(nx, ny)
        if compute_indexes is None:
            compute_indexes = np.ones_like(self._x_grid, dtype=bool)
        supersampled_indexes1d = util.image2array(supersampling_indexes)
        self._high_res_indexes1d = (supersampled_indexes1d) & (compute_indexes)
        self._low_res_indexes1d = (supersampled_indexes1d is False) & (compute_indexes)
        self._supersampling_factor = supersampling_factor
        self._num_sub = supersampling_factor * supersampling_factor
        self._x_low_res = self._x_grid[self._low_res_indexes1d]
        self._y_low_res = self._y_grid[self._low_res_indexes1d]
        self._num_low_res = len(self._x_low_res)

    @property
    def coordinates_evaluate(self):
        """

        :return: 1d array of all coordinates being evaluated to perform the image computation
        """
        ra_low, dec_low = self._low_res_coordinates
        ra_high, dec_high = self._high_res_coordinates
        ra_joint = np.append(ra_low, ra_high)
        dec_joint = np.append(dec_low, dec_high)
        return ra_joint, dec_joint

    def flux_array2image_low_high(self, flux_array):
        """

        :param flux_array: 1d array of low and high resolution flux values corresponding to the coordinates_evaluate
        order
        :return: 2d array, 2d array, corresponding to (partial) images in low and high resolution (to be convolved)
        """
        low_res_values = flux_array[0:self._num_low_res]
        high_res_values = flux_array[self._num_low_res:]
        image_low_res = self._merge_low_high_res(low_res_values, high_res_values)
        image_high_res_partial = self._high_res_image(high_res_values)
        return image_low_res, image_high_res_partial

    @property
    def _high_res_coordinates(self):
        """

        :return: 1d arrays of subpixel grid coordinates
        """
        if not hasattr(self, '_x_sub_grid'):
            self._subpixel_coordinates()
        return self._x_high_res, self._y_high_res

    @property
    def _low_res_coordinates(self):
        """

        :return:
        """
        return self._x_low_res, self._y_low_res

    def _merge_low_high_res(self, low_res_values, supersampled_values):
        """
        adds/overwrites the supersampled values on the image

        :param low_res_values: 1d array of image with low resolution
        :param supersampled_values: values of the supersampled sub-pixels
        :return: 2d image
        """

        array = np.zeros_like(self._x_grid)
        array[self._low_res_indexes1d] = low_res_values
        array_low_res_partial = self._average_subgrid(supersampled_values)
        array[self._high_res_indexes1d] = array_low_res_partial
        return util.array2image(array)

    def _high_res_image(self, supersampled_values):
        """

        :param supersampled_values: 1d array of supersampled values corresponding to coordinates
        :return: 2d array of supersampled image (zeros outside supersampled frame)
        """
        high_res = np.zeros((self._nx * self._supersampling_factor, self._ny * self._supersampling_factor))
        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                selected = supersampled_values[count::self._num_sub]
                high_res[i::self._supersampling_factor, j::self._supersampling_factor] = self._array2image_subset(
                    selected)
                count += 1
        return high_res

    def _subpixel_coordinates(self):
        """

        :return: 1d arrays of subpixel grid coordinates
        """
        x_grid_select = self._x_grid[self._high_res_indexes1d]
        y_grid_select = self._y_grid[self._high_res_indexes1d]
        x_sub_grid = np.zeros(len(x_grid_select) * self._num_sub)
        y_sub_grid = np.zeros(len(x_grid_select) * self._num_sub)
        count = 0
        for i in range(self._supersampling_factor):
            for j in range(self._supersampling_factor):
                x_ij = 1. / self._supersampling_factor / 2. + i / float(self._supersampling_factor) - 1. / 2
                y_ij = 1. / self._supersampling_factor / 2. + j / float(self._supersampling_factor) - 1. / 2
                delta_ra, delta_dec = self.map_pix2coord(x_ij, y_ij)
                delta_ra0, delta_dec_0 = self.map_pix2coord(0, 0)
                x_sub_grid[count::self._num_sub] = x_grid_select + delta_ra - delta_ra0
                y_sub_grid[count::self._num_sub] = y_grid_select + delta_dec - delta_dec_0
                count += 1
        self._x_high_res = x_sub_grid
        self._y_high_res = y_sub_grid

    def _average_subgrid(self, subgrid_values):
        """
        averages the values over a pixel

        :param subgrid_values: values (e.g. flux) of subgrid coordinates
        :return: 1d array of size of the supersampled pixels
        """
        values_2d = np.reshape(subgrid_values, (-1, self._num_sub))
        return np.mean(values_2d, axis=1)

    def _array2image_subset(self, array):
        """
        maps a 1d array into a (nx, ny) 2d grid with array populating the idex_mask indices
        :param array: 1d array
        :return: 2d array
        """
        grid1d = np.zeros(self._nx * self._ny)
        grid1d[self._high_res_indexes1d] = array
        grid2d = util.array2image(grid1d, self._nx, self._ny)
        return grid2d

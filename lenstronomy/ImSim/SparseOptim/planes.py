__author__ = 'aymgal'

import numpy as np
from scipy.ndimage import morphology

from lenstronomy.Util import util


class BasePlaneGrid(object):

    """Base class for image and source plane grids"""

    def __init__(self, data_class):
        num_pix_x, num_pix_y = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width

    @property
    def num_pix(self):
        return self._num_pix

    @property
    def grid_size(self):
        return self._num_pix**2

    @property
    def grid_shape(self):
        return (self._num_pix, self._num_pix)

    @property
    def delta_pix(self):
        return self._delta_pix

    @property
    def theta_x(self):
        if not hasattr(self, '_x_grid_1d'):
            raise ValueError("theta coordinates are not defined")
        return self._x_grid_1d

    @property
    def theta_y(self):
        if not hasattr(self, '_y_grid_1d'):
            raise ValueError("theta coordinates are not defined")
        return self._y_grid_1d

    @property
    def unit_image(self):
        return np.ones(self.grid_shape)


class ImagePlaneGrid(BasePlaneGrid):

    """Class that defines the grid on which lens galaxy is projected"""

    def __init__(self, data_class):
        super(ImagePlaneGrid, self).__init__(data_class)
        # get the coordinates arrays of image plane
        x_grid, y_grid = data_class.pixel_coordinates
        self._x_grid_1d = util.image2array(x_grid)
        self._y_grid_1d = util.image2array(y_grid)


class SourcePlaneGrid(BasePlaneGrid):

    """Class that defines the grid on which source galaxy is projected"""

    def __init__(self, data_class, subgrid_res=1):
        super(SourcePlaneGrid, self).__init__(data_class)
        self._subgrid_res = subgrid_res
        # adapt grid size and resolution
        self._num_pix *= int(subgrid_res)
        self._delta_pix /= float(subgrid_res)
        # get the coordinates arrays of source plane
        self._x_grid_1d, self._y_grid_1d = util.make_grid(numPix=self._num_pix, 
                                                          deltapix=self._delta_pix)

    @property
    def effective_mask(self):
        """
        Returns the intersection between the likelihood mask and the area on source plane
        that has corresponding pixels in image plane
        """
        if not hasattr(self, '_effective_mask'):
            print("Warning : lensed unit image in source plane has not been set, effective mask filled with 1s")
            self._effective_mask = np.ones(self.grid_shape)
        return self._effective_mask.astype(float)

    @property
    def reduction_mask(self):
        if not hasattr(self, '_reduc_mask_1d'):
            print("Warning : no reduction mask has been computed for grid shrinking")
            self._reduc_mask_1d = np.ones(self.grid_size, dtype=bool)
        return util.array2image(self._reduc_mask_1d.astype(float))

    def set_delensed_masks(self, unit_image, mask=None):
        """input unit_image and mask must be non-boolean 2d arrays"""
        image_refined = self._fill_mapping_holes(unit_image).astype(bool)
        if mask is not None:
            mask_refined = self._fill_mapping_holes(mask).astype(bool)
            self._effective_mask = np.zeros(self.grid_shape, dtype=bool)
            self._effective_mask[image_refined & mask_refined] = True
        else:
            self._effective_mask = image_refined

    def shrink_grid_to_mask(self, min_num_pix=None):
        if self.effective_mask is None:
            # if no mask to shrink to, do nothing
            return
        if min_num_pix is None:
            # kind of arbitrary as a default
            min_num_pix = int(self.num_pix / 4)
        reduc_mask, reduced_num_pix = self._reduce_plane_iterative(self.effective_mask, min_num_pix=min_num_pix)
        self._reduc_mask_1d = util.image2array(reduc_mask).astype(bool)
        # backup the original 'large' grid
        self._num_pix_large = self._num_pix
        self._x_grid_1d_large = np.copy(self._x_grid_1d)
        self._y_grid_1d_large = np.copy(self._y_grid_1d)
        self._effective_mask_large = np.copy(self.effective_mask)
        # update coordinates array
        self._num_pix = reduced_num_pix
        self._x_grid_1d = self._x_grid_1d[self._reduc_mask_1d]
        self._y_grid_1d = self._y_grid_1d[self._reduc_mask_1d]
        # don't know why, but can apply reduc_mask_1d only on 1D arrays
        effective_mask_1d = util.image2array(self._effective_mask)
        self._effective_mask = util.array2image(effective_mask_1d[self._reduc_mask_1d])
        print("Source grid has been reduced from {} to {} side pixels".format(self._num_pix_large, self._num_pix))

    def project_on_original_grid(self, image):
        array_large = np.zeros(self._num_pix_large**2)
        array_large[self._reduc_mask_1d] = util.image2array(image)[:]
        return util.array2image(array_large)

    def _fill_mapping_holes(self, image):
        """
        erosion operation for filling holes

        The higher the subgrid resolution of the source, the highest the number of holes.
        Hence the 'strength' of the erosion is set to the subgrid resolution of the source plane 
        """
        strength = self._subgrid_res
        # invert 0s and 1s
        image = 1 - image
        # apply morphological erosion operation
        image = morphology.binary_erosion(image, iterations=strength)
        # invert 1s and 0s
        image = 1 - image
        # remove margins that were
        image[:strength, :] = 0
        image[-strength:, :] = 0
        image[:, :strength] = 0
        image[:, -strength:] = 0
        return image

    def _reduce_plane_iterative(self, effective_mask, min_num_pix=10):
        num_pix = len(effective_mask)  # start at original size
        min_num_pix = 10  # minimal allowed number of pixels in source plane
        n_rm = 1
        test_mask = np.zeros((num_pix, num_pix))
        while num_pix > min_num_pix:
            # array full of zeros
            test_mask_next = np.zeros_like(effective_mask)
            # fill with ones to create a centered square with size reduced by 2*n_rm
            test_mask_next[n_rm:-n_rm, n_rm:-n_rm] = 1
            # update number side length of the non-zero
            num_pix_next = self.num_pix - 2 * n_rm
            # test if all ones in test_mask are also ones in target mask
            intersection_mask = np.zeros_like(test_mask_next)
            intersection_mask[(test_mask_next == 1) & (effective_mask == 1)] = 1
            is_too_large_mask = np.all(intersection_mask == effective_mask)
            if is_too_large_mask:
                # if the intersection is equal to the original array mask, this means that we can try a smaller mask
                num_pix = num_pix_next
                test_mask = test_mask_next
                n_rm += 1
            else:
                # if not, then the mask at previous iteration was the correct one
                reduc_mask = test_mask
                red_num_pix = num_pix
                break
        return reduc_mask.astype(bool), red_num_pix

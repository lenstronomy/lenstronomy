__author__ = 'aymgal'

# class that implements the computation of the lensing operator
# that maps image plane pixels to source plane pixels

import numpy as np
from scipy import sparse

from lenstronomy.ImSim.SparseOptim.planes import ImagePlaneGrid, SourcePlaneGrid
from lenstronomy.Util import util



class LensingOperator(object):

    """TODO"""

    def __init__(self, data_class, lens_model_class, subgrid_res_source=1, 
                 likelihood_mask=None, minimal_source_plane=True, matrix_prod=True):
        self.lensModel = lens_model_class
        self.imagePlane  = ImagePlaneGrid(data_class)
        self.sourcePlane = SourcePlaneGrid(data_class, subgrid_res=subgrid_res_source)
        self._subgrid_res_source = subgrid_res_source
        self._likelihood_mask = likelihood_mask
        self._minimal_source_plane = minimal_source_plane
        self._matrix_prod = matrix_prod

    def source2image(self, source_1d, kwargs_lens=None, update=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self.update_mapping(kwargs_lens)

        if self._matrix_prod:
            image_1d = self._source2image_matrix(source_1d)
        else:
            image_1d = np.ones(self.imagePlane.grid_size)
            # loop over source plane pixels
            for j in range(source_1d.size):
                indices_i = np.where(self._lens_mapping_list == j)
                image[indices_i] = source_1d[j]
        return image_1d

    def _source2image_matrix(self, source_1d):
        image_1d = self._lens_mapping_matrix.dot(source_1d)
        return image_1d

    def source2image_2d(self, source, **kwargs):
        source_1d = util.image2array(source)
        return util.array2image(self.source2image(source_1d, **kwargs))

    def image2source(self, image_1d, kwargs_lens=None, update=False, test_unit_image=False):
        """if test_unit_image is True, do not normalize light flux to better visualize the mapping"""
        if not hasattr(self, '_lens_mapping_list') or update:
            self.update_mapping(kwargs_lens)
        source_1d = np.zeros(self.sourcePlane.grid_size)
        # loop over source plane pixels
        for j in range(source_1d.size):
            # retieve corresponding pixels in image plane
            indices_i = np.where(self._lens_mapping_list == j)
            flux_i = image_1d[indices_i]
            if test_unit_image:
                norm_j = 1
            else:
                norm_j = max(1, flux_i.size)
            flux_j = np.sum(flux_i) / norm_j
            source_1d[j] = flux_j
        return source_1d

    def image2source_2d(self, image, **kwargs):
        image_1d = util.image2array(image)
        return util.array2image(self.image2source(image_1d, **kwargs))

    def lens_mapping_list(self, kwargs_lens=None, update=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self.update_mapping(kwargs_lens)
        return self._lens_mapping_list

    def lens_mapping_matrix(self, kwargs_lens=None, update=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self.update_mapping(kwargs_lens)
        return self._lens_mapping_matrix

    @property
    def source_plane_coordinates(self):
        return self.sourcePlane.theta_x, self.sourcePlane.theta_y

    @property
    def image_plane_coordinates(self):
        return self.imagePlane.theta_x, self.imagePlane.theta_y

    def update_mapping(self, kwargs_lens):
        self._compute_mapping(kwargs_lens)
        self._compute_source_mask()
        if self._minimal_source_plane:
            # for source plane to be reduced to minimal size
            # we compute effective source mask and shrink the grid to match it
            self._shrink_source_plane()
            # recompute the mapping with updated grid
            self._compute_mapping(kwargs_lens)

    def _compute_mapping(self, kwargs_lens):
        """
        Core method that computes the mapping between image and source planes pixels
        from ray-tracing performed by the input parametric mass model
        """
        if self._matrix_prod:
            lens_mapping_array = np.zeros((self.imagePlane.grid_size, self.sourcePlane.grid_size))
        lens_mapping_list = []

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self.lensModel.ray_shooting(self.imagePlane.theta_x, self.imagePlane.theta_y, kwargs_lens)

        # 1. iterate through indices of image plane (indices 'i')
        for i in range(self.imagePlane.grid_size):
            # 2. get the coordinates of ray traced pixels (in image plane coordinates)
            beta_i_x = beta_x[i]
            beta_i_y = beta_y[i]
            
            # 3. compute the closest coordinates in source plane (indices 'j')
            distance_on_source_grid_x = np.abs(beta_i_x - self.sourcePlane.theta_x)
            distance_on_source_grid_y = np.abs(beta_i_y - self.sourcePlane.theta_y)
            j_x = np.argmin(distance_on_source_grid_x)
            j_y = np.argmin(distance_on_source_grid_y)
            #j_x1 = int( beta_i_x / delta_pix + (num_pix - 1)/2.)
            #j_y1 = int( beta_i_y / delta_pix + (num_pix - 1)/2.)
            
            if isinstance(j_x, list):
                j_x = j_x[0]
                print("Warning : found > 1 possible x coordinates in source plane for index i={}".format(i))
            if isinstance(j_y, list):
                j_y = j_y[0]
                print("Warning : found > 1 possible y coordinates in source plane for index i={}".format(i))
            
            #theta_x_j = theta_x_src[j_x]
            #theta_y_j = theta_y_src[j_y]
            
            # 4. find the 1D index that corresponds to these coordinates
            # TODO : find more 'correct' way to find the index j
            j = j_x + j_y   # WRONG ??  but it seems to work
            #print(j_x1 + j_y1, j_x + j_y)
            
            # fill the mapping array
            lens_mapping_list.append(j)
            lens_mapping_array[i, j] = 1

        # convert the list to array 
        self._lens_mapping_list = np.array(lens_mapping_list)
        
        if self._matrix_prod:
            # convert numpy array to sparse matrix, using Compressed Sparse Row (CSR) format for fast vector products
            self._lens_mapping_matrix = sparse.csr_matrix(lens_mapping_array)
            del lens_mapping_array
        else:
            self._lens_mapping_matrix = None

    def _compute_source_mask(self):
        # de-lens a unit image it to get non-zero source plane pixel
        unit_mapped = self.image2source_2d(self.imagePlane.unit_image)
        unit_mapped[unit_mapped > 0] = 1
        if self._likelihood_mask is not None:
            # de-lens a unit image it to get non-zero source plane pixel
            mask_mapped = self.image2source_2d(self._likelihood_mask)
            mask_mapped[mask_mapped > 0] = 1
        else:
            mask_mapped = None
        # set the image to source plane for filling holes due to lensing
        self.sourcePlane.set_delensed_masks(unit_mapped, mask=mask_mapped)

    def _shrink_source_plane(self):
        self.sourcePlane.shrink_grid_to_mask()

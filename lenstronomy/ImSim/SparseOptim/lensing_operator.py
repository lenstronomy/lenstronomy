# class that implements the computation of the lensing operator
# that maps image plane pixels to source plane pixels

import numpy as np
from scipy import sparse

from lenstronomy.Util import util



class LensingOperator(object):

    """TODO"""

    def __init__(self, data_class, lens_model_class, subgrid_res_source=1, matrix_prod=True):
        self._lens_model = lens_model_class
        self._subgrid_res_source = subgrid_res_source
        self._matrix_prod = matrix_prod

        num_pix_x, num_pix_y = data_class.num_pixel_axes
        if num_pix_x != num_pix_y:
            raise ValueError("Only square images are supported")
        self._num_pix = num_pix_x
        self._delta_pix = data_class.pixel_width

        # get the coordinates arrays of image plane
        x_grid, y_grid = data_class.pixel_coordinates
        x_grid_1d = util.image2array(x_grid)
        y_grid_1d = util.image2array(y_grid)
        self.theta_x = x_grid_1d
        self.theta_y = y_grid_1d

        # get the coordinates arrays of source plane
        x_grid_src_1d, y_grid_src_1d = util.make_grid(numPix=self._num_pix, deltapix=self._delta_pix, 
                                                      subgrid_res=self._subgrid_res_source)
        self.theta_x_src = x_grid_src_1d
        self.theta_y_src = y_grid_src_1d


    def source2image(self, source_1d, kwargs_lens, update=False): #, compute_norm=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self._compute_mapping(kwargs_lens)

        # if not compute_norm:
        #     source_ones_1d = np.ones_like(source_1d)
        #     image_ones_1d = self.source2image(source_ones_1d, kwargs_lens, compute_norm=True)
        #     image_ones_1d[image_ones_1d == 0.] = 1.
        #     norm_1d = image_ones_1d
        # else:
        #     norm_1d = 1.

        if self._matrix_prod:
            image_1d = self._source2image_matrix(source_1d, kwargs_lens)
        else:
            image_1d = np.ones(self._num_pix**2)
            for j in range(source_1d.size):
                indices_i = np.where(self._lens_mapping_list == j)
                image[indices_i] = source_1d[j]
        
        # image_1d /= norm_1d
        return image_1d


    def _source2image_matrix(self, source_1d, kwargs_lens):
        image_1d = self._lens_mapping_matrix.dot(source_1d)
        return image_1d


    def image2source(self, image_1d, kwargs_lens, update=False, test_unit_image=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self._compute_mapping(kwargs_lens)
        source_1d = np.zeros(self._num_pix**2 * self._subgrid_res_source**2)
        for j in range(source_1d.size):
            indices_i = np.where(self._lens_mapping_list == j)
            flux_i = image_1d[indices_i]
            if test_unit_image:
                norm_j = 1
            else:
                norm_j = max(1, flux_i.size)
            flux_j = np.sum(flux_i) / norm_j
            source_1d[j] = flux_j
        return source_1d


    def lens_mapping_list(self, kwargs_lens, update=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self._compute_mapping(kwargs_lens)
        return self._lens_mapping_list


    def lens_mapping_matrix(self, kwargs_lens, update=False):
        if not hasattr(self, '_lens_mapping_list') or update:
            self._compute_mapping(kwargs_lens)
        return self._lens_mapping_matrix


    @property
    def source_plane_coordinates(self):
        return self.theta_x_src, self.theta_y_src


    @property
    def image_plane_coordinates(self):
        return self.theta_x, self.theta_y


    def _compute_mapping(self, kwargs_lens):
        image_plane_size  = self._num_pix**2
        source_plane_size = self._num_pix**2 * self._subgrid_res_source**2
        if self._matrix_prod:
            lens_mapping_array = np.zeros((image_plane_size, source_plane_size))
        lens_mapping_list = []

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self._lens_model.ray_shooting(self.theta_x, self.theta_y, kwargs_lens)

        # 1. iterate through indices of image plane (indices 'i')
        for i in range(image_plane_size):
            # 2. get the coordinates of ray traced pixels (in image plane coordinates)
            beta_x_i = beta_x[i]
            beta_y_i = beta_y[i]
            
            # 3. compute the closest coordinates in source plane (indices 'j')
            distance_on_source_grid_x = np.abs(beta_x_i - self.theta_x_src)
            distance_on_source_grid_y = np.abs(beta_y_i - self.theta_y_src)
            j_x = np.argmin(distance_on_source_grid_x)
            j_y = np.argmin(distance_on_source_grid_y)
            #j_x1 = int( beta_x_i / delta_pix + (num_pix - 1)/2.)
            #j_y1 = int( beta_y_i / delta_pix + (num_pix - 1)/2.)
            
            if isinstance(j_x, list):
                j_x = j_x[0]
                print("Warning : found > 1 possible x coordinates in source plane for index i={}".format(i))
            if isinstance(j_y, list):
                j_y = j_y[0]
                print("Warning : found > 1 possible y coordinates in source plane for index i={}".format(i))
            
            #theta_x_j = theta_x_src[j_x]
            #theta_y_j = theta_y_src[j_y]
            
            # 4. find the 1D index that corresponds to these coordinates
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



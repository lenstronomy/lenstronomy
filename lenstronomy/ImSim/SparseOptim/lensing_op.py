# class that implements the computation of the lensing operator
# that maps image plane pixels to source plane pixels

import numpy
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

        # get the coordinates arrays of image plane (the 'thetas')
        x_grid, y_grid = data_class.pixel_coordinates
        x_grid_1d = util.image2array(x_grid)
        y_grid_1d = util.image2array(y_grid)
        self.theta_x = x_grid_1d
        self.theta_y = y_grid_1d

        # get the coordinates arrays of image plane (the 'thetas' in source plane)
        x_grid_src_1d, y_grid_src_1d = lenstro_util.make_grid(numPix=self._num_pix, deltapix=self._delta_pix, 
                                                              subgrid_res=self._subgrid_res_source)
        self.theta_x_src = x_grid_src_1d
        self.theta_y_src = y_grid_src_1d


    def image2source(self, image_1d, kwargs_lens):
        if not hasattr(self, '_lens_mapping_list'):
            self._lens_mapping_list, self._lens_mapping_matrix = self._compute_mapping(kwargs_lens)


    def source2image(self, source_1d, kwargs_lens):
        if not hasattr(self, '_lens_mapping_list'):
            self._lens_mapping_list, self._lens_mapping_matrix = self._compute_mapping(kwargs_lens)
        if self._matrix_prod:
            image_1d = self._lens_mapping_matrix.dot(source_1d)
        else:
            image_1d = np.ones(self._num_pix**2)
            for j in range(source.size):
                i_indices = np.where(mass_mapping == j)
                image[i_indices] = source[j]
        return image_1d


    @property
    def source_plane_coordinates(self):
        return self.theta_x_src, self.theta_y_src


    @property
    def image_plane_coordinates(self):
        return self.theta_x, self.theta_y


    def _compute_mapping(self, kwargs_lens):
        # different ways to store the mapping
        lens_mapping_list = []

        image_plane_size  = self._num_pix**2
        source_plane_size = self._num_pix**2 * self._subgrid_res_source**2
        if self._matrix_prod:
            lens_mapping_array = np.zeros((image_plane_size, source_plane_size))

        # backward ray-tracing to get source coordinates in image plane (the 'betas')
        beta_x, beta_y = self._lens_model.ray_shooting(self.theta_x, self.theta_y, kwargs_lens)

        # 1. iterate through indices of image plane (indices 'i')
        for i in range(image_sim.size):
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
        lens_mapping_list = np.array(lens_mapping_list)
        
        if self._matrix_prod:     
            # convert numpy array to sparse matrix, using Compressed Sparse Row (CSR) format for fast vector products
            lens_mapping_matrix = sparse.csr_matrix(lens_mapping_array)
            del lens_mapping_array
        else:
            lens_mapping_matrix = None

        return lens_mapping_list, lens_mapping_matrix

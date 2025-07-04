import numpy as np
from tqdm import tqdm

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.util as util

__all__ = ['PixelatedSourceReconstruction']

class PixelatedSourceReconstruction(object):
    """
    This class defines useful functions for pixelated source plane reconstruction in gravitational lensing.
    It handles initialization of data, lens model, and PSF kernel, and provides methods for generating
    the M and b matrices for source reconstruction based on different likelihood methods.
    """
    
    def __init__(self, kwargs_data, lens_model_list, kwargs_lens, kernel, verbose=False):
        """
        :param kwargs_data: keyword arguments for `lenstronomy.Data.imaging_data.ImageData`.
            Expected to contain image data, pixel scale, noise, coordinates, 
            and optional primary beam/likelihood method (for inteferometric natwt likelihood method).
        :param lens_model_list: List of lens model types (e.g., ['SIE', 'SHEAR']).
        :param kwargs_lens: List of keyword arguments for each lens model in `lens_model_list`.
        :param kernel: 2D numpy array representing the Point Spread Function (PSF) kernel.
            Must be square, have odd dimensions, and be sufficiently large (>= 2 * numPix - 1).
        :param verbose: If True, print progress messages during matrix generation steps. Defaults to False.
        :type verbose: bool
        
        :raises ValueError: If the PSF kernel is improperly sized or if an unsupported likelihood method is specified.
        """
        lens_model_class = LensModel(lens_model_list)
        data_class = ImageData(**kwargs_data)
        
        self._numPix = len(kwargs_data['image_data'])  # Number of pixels along one dimension of the image
        self._deltaPix = kwargs_data['transform_pix2angle'][0,0] # Angular size of a single pixel
        self._image_data = kwargs_data['image_data']  # Observed image data
        self._noise_rms = kwargs_data['background_rms'] # RMS noise of the background
        
        # Validate PSF kernel size
        shape_kernel_cut = kernel.shape
        for check_dim in range(2):
            if shape_kernel_cut[check_dim] % 2 == 0 or shape_kernel_cut[check_dim] < 2 * self._numPix - 1:
                raise ValueError('PSF kernel size must be odd and at least (2 * numPix - 1) '
                                 'in each dimension for proper convolution.')
        self._kernel = kernel
        
        # (RA, DEC) coordinate at pixel (0,0) of the image plane in angular units
        self._minx = kwargs_data['ra_at_xy_0']
        self._miny = kwargs_data['dec_at_xy_0']
        
        # Calculate lensed source plane coordinates for each image pixel
        x_grid_temp, y_grid_temp = data_class.pixel_coordinates
        self._x_grid = util.image2array(x_grid_temp)
        self._y_grid = util.image2array(y_grid_temp)
        self._beta_x_grid, self._beta_y_grid = lens_model_class.ray_shooting(self._x_grid, self._y_grid, kwargs=kwargs_lens)
        
        self._primary_beam = kwargs_data.get('antenna_primary_beam', None) # Optional primary beam map
        self._logL_method = kwargs_data.get('likelihood_method', "diagonal") # Likelihood method for M and b generation
        self._verbose = verbose

        if self._logL_method not in ["diagonal", "interferometry_natwt"]:
            raise ValueError(
                "likelihood_method '%s' not supported! It can only be 'diagonal' or 'interferometry_natwt'."
                % self._logL_method
            )
        
    def generate_M_b(self, x_min, x_max, y_min, y_max):
        """
        Generates the M and b matrices for source reconstruction based on the selected likelihood method.

        The parameters `x_min`, `x_max`, `y_min`, `y_max` define a **rectangular sub-region of the source plane**
        for the reconstruction. These are integer pixel indices, starting from `0`, relative to the
        image plane's origin (defined by `self._minx`, `self._miny`) and pixel scale (`self._deltaPix`).

        :param x_min: Minimum x-pixel coordinate of the source region (inclusive).
        :param x_max: Maximum x-pixel coordinate of the source region (exclusive).
        :param y_min: Minimum y-pixel coordinate of the source region (inclusive).
        :param y_max: Maximum y-pixel coordinate of the source region (exclusive).
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        :raises TypeError: If `x_min`, `x_max`, `y_min`, `y_max` are not integers.
        :raises ValueError: If coordinates are out of bounds (`< 0` or `> self._numPix`).
        """
        # Input validation for pixel coordinates
        for coord, name in zip([x_min, x_max, y_min, y_max], ['x_min', 'x_max', 'y_min', 'y_max']):
            if not isinstance(coord, int):
                raise TypeError(f"'{name}' must be an integer, but got {type(coord).__name__}.")

        if not (0 <= x_min < x_max <= self._numPix and 0 <= y_min < y_max <= self._numPix):
            raise ValueError(
                f"Source region coordinates out of bounds or invalid order. "
                f"Expected 0 <= x_min < x_max <= {self._numPix} and "
                f"0 <= y_min < y_max <= {self._numPix}. "
                f"Got x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}."
            )
        if self._verbose:
            # Print the total number of source pixels in the defined rectangular source region.
            print("number of source pixels:", (x_max - x_min) * (y_max - y_min))
            print("likelihood method:", self._logL_method)
        if self._logL_method == 'diagonal':
            M, b = self.generate_M_b_diagonal_likelihood(x_min, x_max, y_min, y_max)
        elif self._logL_method == 'interferometry_natwt':
            M, b = self.generate_M_b_interferometry_natwt_likelihood(x_min, x_max, y_min, y_max)
        return M, b
    
    def generate_M_b_diagonal_likelihood(self, x_min, x_max, y_min, y_max):
        """
        Generates M and b matrices assuming spatially uncorrelated noise with uniform RMS across the image.
        This approach is typically used for CCD image data.

        This method performs lensing, convolution, and then computes M and b.

        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if self._verbose:
            print("Step 1: Lensing the source pixels")
        lensed_sp = self.lens_pixel_source_of_a_rectangular_region(x_min, x_max, y_min, y_max)
        if self._verbose:
            print("Step 1: Finished!")

        if self._verbose:
            print("Step 2: Convolve the lensed pixels")
        N_lensed = len(lensed_sp)
        lensed_pixel_conv_set = np.zeros((N_lensed, self._numPix, self._numPix))
        for i in range(N_lensed):
            lensed_pixel_conv_set[i] = self.sparse_convolution(lensed_sp[i], self._kernel)
        if self._verbose:
            print("Step 2: Finished!")

        if self._verbose:
            print("Step 3: Compute the matrix M and vector b")
        M = np.zeros((N_lensed,N_lensed))
        b = np.zeros((N_lensed))

        for i in tqdm(range(N_lensed), desc="Running (iteration times vary)"):
            b[i] = np.sum(lensed_pixel_conv_set[i] * self._image_data)
            for j in range(N_lensed):
                if j < i:
                    M[i,j] = M[j,i] # Exploit symmetry
                else:
                    M[i,j] = np.sum(lensed_pixel_conv_set[i] * lensed_pixel_conv_set[j])

        b /= self._noise_rms**2
        M /= self._noise_rms**2
        if self._verbose:
            print("Step 3: Finished!")

        return M, b
    
    def generate_M_b_interferometry_natwt_likelihood(self, x_min, x_max, y_min, y_max):
        """
        Generates the M and b matrices for interferometric data with natural weighting.

        This method integrates the convolution step implicitly via specialized sparse product functions.

        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if self._verbose:
            print("Step 1: Lensing the source pixels")
        lensed_sp = self.lens_pixel_source_of_a_rectangular_region(x_min, x_max, y_min, y_max)
        if self._verbose:
            print("Step 1: Finished!")
        
        if self._verbose:
            print("Step 2: Compute the matrix M and vector b (including the convolution step)")
        N_lensed = len(lensed_sp)
        M = np.zeros((N_lensed,N_lensed))
        b = np.zeros((N_lensed))
        
        for i in tqdm(range(N_lensed), desc="Running (iteration times vary)"):
            b[i] = self.sum_sparse_elementwise_product(lensed_sp[i], self._image_data)
            pixel_lensed_convolved = self.sparse_convolution(lensed_sp[i], self._kernel) # convolution
            for j in range(N_lensed):
                if j < i:
                    M[i,j] = M[j,i] # Exploit symmetry
                else:
                    M[i,j] = self.sum_sparse_elementwise_product(lensed_sp[j], pixel_lensed_convolved)
        
        b /= self._noise_rms**2
        M /= self._noise_rms**2
        if self._verbose:
            print("Step 2: Finished!")
        
        return M, b
        
    def lens_pixel_source_of_a_rectangular_region(self, x_min, x_max, y_min, y_max):
        """
        Maps image plane pixels to source plane pixels within a specified rectangular region,
        considering lensing deflections and applying bilinear interpolation.

        :returns: A list of lists. Each element in the outer list corresponds to a single source pixel
            within the defined sub-grid (ordered by their linear index). Each inner list contains
            `[image_x_idx, image_y_idx, weight]` tuples, indicating that the source pixel at this index
            contributes with `weight` to the image plane pixel at `(image_x_idx, image_y_idx)` when lensed.
        :rtype: list
        """
        xlen = x_max - x_min
        ylen = y_max - y_min
        
        lensed_pixel_sp = [[] for _ in range(xlen * ylen)]
    
        beta_x_grid_2d = util.array2image(self._beta_x_grid)
        beta_y_grid_2d = util.array2image(self._beta_y_grid)
        
        # Calculate integer pixel indices (floor/ceiling) in the source plane
        x_floor = ((beta_x_grid_2d - self._minx) / self._deltaPix).astype(int)
        x_ceiling = x_floor + 1
        y_floor = ((beta_y_grid_2d - self._miny) / self._deltaPix).astype(int)
        y_ceiling = y_floor + 1
    
        # Calculate fractional pixel offsets for bilinear interpolation
        delta_x_pixel = (beta_x_grid_2d - self._minx) / self._deltaPix - x_floor
        delta_y_pixel = (beta_y_grid_2d - self._miny) / self._deltaPix - y_floor
    
        # Compute bilinear interpolation weights
        w00 = (1 - delta_x_pixel) * (1 - delta_y_pixel)
        w10 = delta_x_pixel * (1 - delta_y_pixel)
        w01 = delta_y_pixel * (1 - delta_x_pixel)
        w11 = delta_x_pixel * delta_y_pixel
    
        # Apply primary beam modulation if specified
        if self._primary_beam is not None:
            w00 *= self._primary_beam
            w10 *= self._primary_beam
            w01 *= self._primary_beam
            w11 *= self._primary_beam
    
        for i in range(self._numPix):
            for j in range(self._numPix):
                # Skip if the lensed image pixel falls entirely outside the source region
                if (x_ceiling[i][j] < x_min or x_floor[i][j] > (x_max - 1) or 
                    y_ceiling[i][j] < y_min or y_floor[i][j] > (y_max - 1)):
                    continue
                
                # Calculate linear indices of the four potential source pixels
                source_pixel_index00 = xlen * (y_floor[i][j] - y_min) + x_floor[i][j] - x_min
                source_pixel_index10 = source_pixel_index00 + 1
                source_pixel_index01 = source_pixel_index00 + xlen
                source_pixel_index11 = source_pixel_index01 + 1
    
                # Flags for boundary checking
                check00, check01, check10, check11 = True, True, True, True
                
                # Adjust flags for contributions near the boundary
                if x_floor[i][j] == x_min - 1:
                    check00 = False
                    check01 = False
                elif x_ceiling[i][j] == x_max:
                    check10 = False
                    check11 = False
    
                if y_floor[i][j] == y_min - 1:
                    check00 = False
                    check10 = False
                elif y_ceiling[i][j] == y_max:
                    check11 = False
                    check01 = False
                
                # Append the contribution of each source pixel to the image pixel (i,j) with weight.
                if check00:
                    lensed_pixel_sp[source_pixel_index00].append([i,j,w00[i][j]])
                if check10:
                    lensed_pixel_sp[source_pixel_index10].append([i,j,w10[i][j]])
                if check01:
                    lensed_pixel_sp[source_pixel_index01].append([i,j,w01[i][j]])
                if check11:
                    lensed_pixel_sp[source_pixel_index11].append([i,j,w11[i][j]])
        return lensed_pixel_sp


    def lens_an_image_by_rayshooting(self, image):
        """
        Lenses a pixelated source plane image to the image plane using ray-shooting and bilinear interpolation.
    
        This method works by iterating through each pixel in the image plane, ray-shooting back to the source
        plane to find the corresponding source coordinate, and then interpolating the flux from the input
        source image at that coordinate. This provides an approximate lensed image, as flux between exact
        source pixels is derived via interpolation. Note that the primary beam will NOT be applied on the 
        lensed image in this function.
    
        :param image: 2D NumPy array representing the pixelated source plane image. Expected to have
                      dimensions (`self._numPix`, `self._numPix`).
        :type image: numpy.ndarray
        :returns: 2D NumPy array representing the lensed image in the image plane.
        :rtype: numpy.ndarray
        :raises ValueError: If the input `image` dimensions do not match `self._numPix`.
        """
        nx, ny = np.shape(image)
        if nx != self._numPix or ny != self._numPix:
            raise ValueError(f"Input image size ({nx}, {ny}) must match the defined "
                             f"data class dimension ({self._numPix}, {self._numPix}).")
            
        lensed_image = np.zeros((nx, ny))
    
        # Iterate through each pixel in the image plane (i, j)
        for i in range(nx):
            for j in range(ny):
                # Calculate the linear index for accessing pre-computed ray-shot coordinates
                n_beta = i * ny + j # Assuming beta_x_grid and beta_y_grid are flattened row-major

                # Get the source plane angular coordinates (beta_x, beta_y) corresponding
                # to the current image plane pixel (i,j) after ray-shooting.
                cor_beta_x = self._beta_x_grid[n_beta]
                cor_beta_y = self._beta_y_grid[n_beta]

                # Convert source plane angular coordinates to integer pixel indices (n_x, n_y)
                # within the source grid, relative to self._minx, self._miny.
                n_x = int((cor_beta_x - self._minx) / self._deltaPix)
                n_y = int((cor_beta_y - self._miny) / self._deltaPix) 
            
                # If the ray shoots outside the defined source image boundaries, set flux to zero
                if n_x > nx - 1 or n_y > ny - 1 or n_x < 0 or n_y < 0:
                    lensed_image[i, j] = 0
                else:
                    # Calculate bilinear interpolation weights for the four surrounding source pixels.
                    # These weights depend on the sub-pixel position of (cor_beta_x, cor_beta_y)
                    # within the source pixel (n_x, n_y).
                    weight_upper_left = np.abs(self._miny + n_y*self._deltaPix + self._deltaPix - cor_beta_y) * (
                        np.abs(self._minx + n_x*self._deltaPix + self._deltaPix - cor_beta_x)) / (self._deltaPix**2) 
                    weight_upper_right = np.abs(self._miny + n_y*self._deltaPix + self._deltaPix - cor_beta_y) * (
                        np.abs(self._minx + n_x*self._deltaPix - cor_beta_x)) / (self._deltaPix**2)
                    weight_lower_left = np.abs(self._minx + n_x*self._deltaPix + self._deltaPix - cor_beta_x) * (
                        np.abs(self._miny + n_y*self._deltaPix - cor_beta_y)) / (self._deltaPix**2)
                    weight_lower_right = np.abs(self._minx + n_x*self._deltaPix - cor_beta_x) * (
                        np.abs(self._miny + n_y*self._deltaPix - cor_beta_y)) / (self._deltaPix**2)
                    
                    # Interpolate flux from the source image using the calculated weights
                    lensed_image[i,j] = image[n_y, n_x] * weight_upper_left
                    if n_x + 1 < nx:
                        lensed_image[i,j] += image[n_y, n_x + 1] * weight_upper_right
                    if n_y + 1 < ny:
                        lensed_image[i,j] += image[n_y + 1, n_x] * weight_lower_left
                        if n_x + 1 < nx:
                            lensed_image[i,j] += image[n_y + 1, n_x + 1] * weight_lower_right
                
        return lensed_image

    def sparse_to_array(self, sparse):
        """
        Converts a sparse image representation (list of `[row_idx, col_idx, value]` tuples) to a 2D NumPy array.

        :param sparse: A list representing non-zero elements of the sparse image.
        :returns: A 2D NumPy array representing the full image.
        :rtype: numpy.ndarray
        """
        image = np.zeros((self._numPix, self._numPix))
        num_of_elements = len(sparse)
        for i in range(num_of_elements):
            image[sparse[i][0], sparse[i][1]] = sparse[i][2]
        return image

    def sum_sparse_elementwise_product(self, sparse, ordinary):
        """
        Computes the element-wise sum of products between a sparse matrix and a dense 2D NumPy array image.

        :param sparse: Sparse matrix representation (list of `[row_idx, col_idx, value]` tuples).
        :param ordinary: A 2D NumPy array (dense matrix).
        :returns: The sum of the element-wise products.
        :rtype: float
        """
        sum_temp = 0
        num_element = len(sparse)
        for i in range(num_element):
            sum_temp += sparse[i][2] * ordinary[sparse[i][0], sparse[i][1]]
        return sum_temp
    
    def sparse_convolve_and_dot_product(self, sp1, sp2, kernel):
        """
        Computes the convolution product of two sparse matrices using a given kernel.
        Equivalent to `(sp1 * kernel) . (sp2)` where '*' is convolution and '.' is dot product.

        :param sp1: First sparse matrix representation.
        :param sp2: Second sparse matrix representation.
        :param kernel: The 2D PSF kernel (assumed odd dimensions, center at central pixel).
        :returns: The result of the convolution product.
        :rtype: float
        """
        inner_product = 0
        num_elements_1 = len(sp1)
        num_elements_2 = len(sp2)
        kernel_center = int(len(kernel) / 2) # Assumes kernel is square and has odd dimensions
        
        for i in range(num_elements_1):
            for j in range(num_elements_2):
                delta_x = sp1[i][0] - sp2[j][0]
                delta_y = sp1[i][1] - sp2[j][1]
                inner_product += sp1[i][2] * sp2[j][2] * kernel[kernel_center + delta_x, kernel_center + delta_y]
        return inner_product

    def sparse_convolution(self, sp, kernel):
        """
        Performs convolution of a sparse matrix with a given kernel.

        :param sp: Sparse matrix representation.
        :param kernel: The 2D PSF kernel (assumed odd dimensions, center at central pixel).
        :returns: A 2D NumPy array representing the convolved image.
        :rtype: numpy.ndarray
        """
        kernel_center = int(len(kernel) / 2) # Assumes kernel is square and has odd dimensions
        convolved = np.zeros((self._numPix, self._numPix))
        num_element_sparse = len(sp)
        
        for i in range(num_element_sparse):
            row_sp, col_sp, val_sp = sp[i][0], sp[i][1], sp[i][2]
            
            # Calculate slice indices for the kernel relative to the sparse element
            slice_x_start = kernel_center - row_sp
            slice_x_end = slice_x_start + self._numPix
            slice_y_start = kernel_center - col_sp
            slice_y_end = slice_y_start + self._numPix
            
            convolved += val_sp * kernel[slice_x_start:slice_x_end, slice_y_start:slice_y_end]
        return convolved
import numpy as np
from tqdm import tqdm

import lenstronomy.Util.util as util

__all__ = ["PixelatedSourceReconstruction"]


class PixelatedSourceReconstruction(object):
    """This class defines useful functions for pixelated source plane reconstruction in
    gravitational lensing. It handles initialization of data, lens model, and PSF
    kernel, and provides methods for generating the M and b matrices (definition see
    'placeholder for Nan Zhang's paper) for source reconstruction based on different
    likelihood methods.

    All sparse matrices (sp) in this class are represented as a list of lists:
    [[y_coord, x_coord, pixel_value], ...], where [y_coord, x_coord] are
    the pixel coordinates and pixel_value is the corresponding pixel's value.
    """

    def __init__(
        self, data_class, psf_class, lens_model_class, source_pixel_grid_class
    ):
        """Initializes the PixelatedSourceReconstruction class. This sets up the
        necessary data, PSF, lens model, and source grid for subsequent source
        reconstruction matrix generation.

        :param data_class: ImageData() class instance (for the observed image data)
        :param psf_class: PSF() class instance (for the observed image data)
        :param lens_model_class: LensModel class instance
        :param source_pixel_grid_class: PixelGrid() class instance (defining the source plane grid)
        :raises ValueError:
            - If the source pixel grid has rotational components or non-uniform pixel widths.
            - If the PSF kernel size is improperly sized for interferometric likelihood methods.
        """

        self._numPix = data_class.num_pixel_axes[0]
        self._image_data = data_class.data
        self._noise_rms = data_class.background_rms
        self._C_D = data_class.C_D
        self._primary_beam = data_class.primary_beam
        self._logL_method = data_class.likelihood_method()

        # prepare for the rayshooting
        self._x_grid_data, self._y_grid_data = data_class.pixel_coordinates
        self._lens_model_class = lens_model_class

        self._source_grid_class = source_pixel_grid_class
        # Validate source grid properties: no rotation and uniform pixel width
        transform_pix2angle_source = self._source_grid_class.transform_pix2angle
        if (
            transform_pix2angle_source[0, 1] != 0
            or transform_pix2angle_source[1, 0] != 0
            or transform_pix2angle_source[0, 0] != transform_pix2angle_source[1, 1]
        ):
            raise ValueError(
                "Source grid must be non-rotational and have uniform pixel width along x and y axes. "
                "Ensure off-diagonal elements of 'transform_pix2angle_source' are zero "
                "and diagonal elements are equal."
            )

        self._nx_source, self._ny_source = self._source_grid_class.num_pixel_axes
        self._num_pixel_source = self._source_grid_class.num_pixel
        self._pixel_width_source = self._source_grid_class.pixel_width
        self._source_min_x, self._source_min_y = self._source_grid_class.radec_at_xy_0
        self._ratio_data_pixel_source_pixel = (
            data_class.pixel_area / source_pixel_grid_class.pixel_area
        )

        self._kernel = psf_class.kernel_point_source
        self._shape_kernel = self._kernel.shape

        # Validate PSF kernel size specifically for interferometric likelihood
        if self._logL_method == "interferometry_natwt":
            for check_dim in range(2):
                if self._shape_kernel[check_dim] < 2 * self._numPix - 1:
                    raise ValueError(
                        "PSF kernel size must be at least (2 * numPix - 1) "
                        "in each dimension for interferometry_natwt likelihood."
                    )

    def generate_M_b(self, kwargs_lens, verbose=False, show_progress=True):
        """Generates the M matrix and the b vector for source reconstruction based on
        the selected likelihood method. Definitions of the M and b is given by
        (placeholder for Nan Zhang's paper).

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param verbose: If True, print progress messages during matrix generation steps.
            Defaults to False.
        :param show_progress: If True, show progress bar of generating the M matrix.
            Defaults to True.
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if verbose:
            # Print the total number of source pixels in the defined rectangular source region.
            print(
                "number of source pixels:",
                self._source_grid_class.num_pixel,
                "(x axis:",
                self._source_grid_class.num_pixel_axes[0],
                "pixels; ",
                "y axis:",
                self._source_grid_class.num_pixel_axes[1],
                "pixels)",
            )
            print("likelihood method:", self._logL_method)
        if self._logL_method == "diagonal":
            M, b = self.generate_M_b_diagonal_likelihood(
                kwargs_lens, verbose, show_progress
            )
        elif self._logL_method == "interferometry_natwt":
            M, b = self.generate_M_b_interferometry_natwt_likelihood(
                kwargs_lens, verbose, show_progress
            )
        return M, b

    def generate_M_b_diagonal_likelihood(
        self, kwargs_lens, verbose=False, show_progress=True
    ):
        """Generates M and b matrices assuming spatially uncorrelated noise with uniform
        RMS across the image. This approach is typically used for CCD image data.

        This method performs lensing, convolution, and then computes M and b.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param verbose: If True, print progress messages during matrix generation steps.
            Defaults to False.
        :param show_progress: If True, show progress bar of generating the M matrix.
            Defaults to True.
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if verbose:
            print("Step 1: Lensing the source pixels")
        lensed_sp = self.lens_pixel_source_of_a_rectangular_region(kwargs_lens)
        if verbose:
            print("Step 1: Finished!")

        if verbose:
            print("Step 2: Convolve the lensed pixels")
        N_lensed = len(lensed_sp)
        lensed_pixel_conv_set = np.zeros((N_lensed, self._numPix, self._numPix))
        for i in range(N_lensed):
            lensed_pixel_conv_set[i] = self.sparse_convolution(
                lensed_sp[i], self._kernel
            )
        if verbose:
            print("Step 2: Finished!")

        if verbose:
            print("Step 3: Compute the matrix M and vector b")
        M = np.zeros((N_lensed, N_lensed))
        b = np.zeros((N_lensed))
        for i in tqdm(
            range(N_lensed),
            desc="Running (iteration times vary)",
            disable=not show_progress,
        ):
            b[i] = np.sum(lensed_pixel_conv_set[i] * self._image_data / self._C_D)
            for j in range(N_lensed):
                if j < i:
                    M[i, j] = M[j, i]  # Exploit symmetry
                else:
                    M[i, j] = np.sum(
                        lensed_pixel_conv_set[i] * lensed_pixel_conv_set[j] / self._C_D
                    )
        if verbose:
            print("Step 3: Finished!")

        return M, b

    def generate_M_b_interferometry_natwt_likelihood(
        self, kwargs_lens, verbose=False, show_progress=True
    ):
        """Generates the M and b matrices for interferometric data with natural
        weighting.

        This method integrates the convolution step implicitly via specialized sparse
        product functions.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param verbose: If True, print progress messages during matrix generation steps.
            Defaults to False.
        :param show_progress: If True, show progress bar of generating the M matrix.
            Defaults to True.
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if verbose:
            print("Step 1: Lensing the source pixels")
        lensed_sp = self.lens_pixel_source_of_a_rectangular_region(kwargs_lens)
        if verbose:
            print("Step 1: Finished!")

        if verbose:
            print(
                "Step 2: Compute the matrix M and vector b (including the convolution step)"
            )
        N_lensed = len(lensed_sp)
        M = np.zeros((N_lensed, N_lensed))
        b = np.zeros((N_lensed))
        for i in tqdm(
            range(N_lensed),
            desc="Running (iteration times vary)",
            disable=not show_progress,
        ):
            b[i] = self.sum_sparse_elementwise_product(lensed_sp[i], self._image_data)
            pixel_lensed_convolved = self.sparse_convolution(
                lensed_sp[i], self._kernel
            )  # convolution
            for j in range(N_lensed):
                if j < i:
                    M[i, j] = M[j, i]  # Exploit symmetry
                else:
                    M[i, j] = self.sum_sparse_elementwise_product(
                        lensed_sp[j], pixel_lensed_convolved
                    )
        b /= self._noise_rms**2
        M /= self._noise_rms**2
        if verbose:
            print("Step 2: Finished!")

        return M, b

    def lens_pixel_source_of_a_rectangular_region(self, kwargs_lens):
        """Maps image plane pixels to source plane pixels within a specified rectangular
        source grid, considering lensing deflections and applying bilinear
        interpolation.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :returns: A list of lists. Each element in the outer list corresponds to a single source pixel
            within the defined source grid (ordered by their linear index). Each inner list contains
            `[y_coord, x_coord, weight]` tuples, indicating that the source pixel at this index
            contributes with `weight` to the image plane pixel at `(y_coord, x_coord)` when lensed.
        :rtype: list
        """

        lensed_pixel_sp = [[] for _ in range(self._num_pixel_source)]

        beta_x_grid_2d, beta_y_grid_2d = self._lens_model_class.ray_shooting(
            self._x_grid_data, self._y_grid_data, kwargs=kwargs_lens
        )

        # Calculate integer pixel indices (floor/ceiling) in the source plane
        x_floor = np.floor(
            (beta_x_grid_2d - self._source_min_x) / self._pixel_width_source
        ).astype(int)
        x_ceiling = x_floor + 1
        y_floor = np.floor(
            (beta_y_grid_2d - self._source_min_y) / self._pixel_width_source
        ).astype(int)
        y_ceiling = y_floor + 1

        # Calculate fractional pixel offsets for bilinear interpolation
        delta_x_pixel = (
            beta_x_grid_2d - self._source_min_x
        ) / self._pixel_width_source - x_floor
        delta_y_pixel = (
            beta_y_grid_2d - self._source_min_y
        ) / self._pixel_width_source - y_floor

        # Compute bilinear interpolation weights
        w00 = (1 - delta_x_pixel) * (1 - delta_y_pixel)
        w10 = delta_x_pixel * (1 - delta_y_pixel)
        w01 = delta_y_pixel * (1 - delta_x_pixel)
        w11 = delta_x_pixel * delta_y_pixel

        # Apply the ratio (data image pixel area / source grid pixel area) to ensure the flux conservation
        w00 *= self._ratio_data_pixel_source_pixel
        w10 *= self._ratio_data_pixel_source_pixel
        w01 *= self._ratio_data_pixel_source_pixel
        w11 *= self._ratio_data_pixel_source_pixel

        # Apply primary beam modulation if specified
        if self._primary_beam is not None:
            w00 *= self._primary_beam
            w10 *= self._primary_beam
            w01 *= self._primary_beam
            w11 *= self._primary_beam

        for i in range(self._numPix):
            for j in range(self._numPix):
                # Skip if the lensed image pixel falls entirely outside the source region
                if (
                    x_ceiling[i][j] < 0
                    or x_floor[i][j] > (self._nx_source - 1)
                    or y_ceiling[i][j] < 0
                    or y_floor[i][j] > (self._ny_source - 1)
                ):
                    continue

                # Calculate linear indices of the four potential source pixels
                source_pixel_index00 = self._nx_source * (y_floor[i][j]) + x_floor[i][j]
                source_pixel_index10 = source_pixel_index00 + 1
                source_pixel_index01 = source_pixel_index00 + self._nx_source
                source_pixel_index11 = source_pixel_index01 + 1

                # Flags for boundary checking
                check00, check01, check10, check11 = True, True, True, True

                # Adjust flags for contributions near the boundary
                if x_floor[i][j] == -1:
                    check00 = False
                    check01 = False
                elif x_ceiling[i][j] == self._nx_source:
                    check10 = False
                    check11 = False

                if y_floor[i][j] == -1:
                    check00 = False
                    check10 = False
                elif y_ceiling[i][j] == self._ny_source:
                    check11 = False
                    check01 = False

                # Append the contribution of each source pixel to the image pixel (i,j) with weight.
                if check00:
                    lensed_pixel_sp[source_pixel_index00].append([i, j, w00[i][j]])
                if check10:
                    lensed_pixel_sp[source_pixel_index10].append([i, j, w10[i][j]])
                if check01:
                    lensed_pixel_sp[source_pixel_index01].append([i, j, w01[i][j]])
                if check11:
                    lensed_pixel_sp[source_pixel_index11].append([i, j, w11[i][j]])
        return lensed_pixel_sp

    def lens_an_image_by_rayshooting(self, kwargs_lens, source_image):
        """Lenses a pixelated source plane image to the image plane using ray-shooting
        and bilinear interpolation. The imput image should have the same dimension and
        coordinates defined by source_pixel_grid_class.

        This method works by iterating through each pixel in the image plane, ray-shooting back to the source
        plane to find the corresponding source coordinate, and then interpolating the flux from the input
        source image at that coordinate. This provides an approximate lensed image, as flux between exact
        source pixels is derived via interpolation. Note that the primary beam will NOT be applied on the
        lensed image in this function.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param image: 2D NumPy array representing the pixelated source plane image. Expected to have
                      dimensions defined by source_pixel_grid_class.
        :type image: numpy.ndarray
        :returns: 2D NumPy array representing the lensed image in the image plane.
        :rtype: numpy.ndarray
        :raises ValueError: If the input `image` dimensions do not match the dimensions of the
                            defined source pixel grid (`self._ny_source`, `self._nx_source`).
        """

        ny_source_check, nx_source_check = np.shape(source_image)
        if nx_source_check != self._nx_source or ny_source_check != self._ny_source:
            raise ValueError(
                f"Input image size ({ny_source_check}, {nx_source_check}) must match the defined "
                f"source grid class dimension ({self._ny_source}, {self._nx_source})."
            )

        lensed_image = np.zeros((self._numPix, self._numPix))

        beta_x_grid_2d, beta_y_grid_2d = self._lens_model_class.ray_shooting(
            self._x_grid_data, self._y_grid_data, kwargs=kwargs_lens
        )
        beta_x_grid = util.image2array(beta_x_grid_2d)
        beta_y_grid = util.image2array(beta_y_grid_2d)

        # Iterate through each pixel in the image plane (i, j)
        for i in range(self._numPix):
            for j in range(self._numPix):
                # Calculate the linear index for accessing pre-computed ray-shot coordinates
                n_beta = (
                    i * self._numPix + j
                )  # Assuming beta_x_grid and beta_y_grid are flattened row-major

                # Get the source plane angular coordinates (beta_x, beta_y) corresponding
                # to the current image plane pixel (i,j) after ray-shooting.
                cor_beta_x = beta_x_grid[n_beta]
                cor_beta_y = beta_y_grid[n_beta]

                # Convert source plane angular coordinates to integer pixel indices (n_y, n_x)
                # within the source grid, relative to self._minx, self._miny.
                n_x = np.floor(
                    (cor_beta_x - self._source_min_x) / self._pixel_width_source
                ).astype(int)
                n_y = np.floor(
                    (cor_beta_y - self._source_min_y) / self._pixel_width_source
                ).astype(int)

                # If the ray shoots outside the defined source image boundaries, set flux to zero
                if (
                    n_x > self._nx_source - 1
                    or n_y > self._ny_source - 1
                    or n_x < 0
                    or n_y < 0
                ):
                    lensed_image[i, j] = 0
                else:
                    # Calculate bilinear interpolation weights for the four surrounding source pixels.
                    # These weights depend on the sub-pixel position of (cor_beta_y, cor_beta_x)
                    # within the source pixel (n_y, n_x).
                    weight_upper_left = (
                        np.abs(
                            self._source_min_y
                            + n_y * self._pixel_width_source
                            + self._pixel_width_source
                            - cor_beta_y
                        )
                        * (
                            np.abs(
                                self._source_min_x
                                + n_x * self._pixel_width_source
                                + self._pixel_width_source
                                - cor_beta_x
                            )
                        )
                        / (self._pixel_width_source**2)
                    )
                    weight_upper_right = (
                        np.abs(
                            self._source_min_y
                            + n_y * self._pixel_width_source
                            + self._pixel_width_source
                            - cor_beta_y
                        )
                        * (
                            np.abs(
                                self._source_min_x
                                + n_x * self._pixel_width_source
                                - cor_beta_x
                            )
                        )
                        / (self._pixel_width_source**2)
                    )
                    weight_lower_left = (
                        np.abs(
                            self._source_min_x
                            + n_x * self._pixel_width_source
                            + self._pixel_width_source
                            - cor_beta_x
                        )
                        * (
                            np.abs(
                                self._source_min_y
                                + n_y * self._pixel_width_source
                                - cor_beta_y
                            )
                        )
                        / (self._pixel_width_source**2)
                    )
                    weight_lower_right = (
                        np.abs(
                            self._source_min_x
                            + n_x * self._pixel_width_source
                            - cor_beta_x
                        )
                        * (
                            np.abs(
                                self._source_min_y
                                + n_y * self._pixel_width_source
                                - cor_beta_y
                            )
                        )
                        / (self._pixel_width_source**2)
                    )

                    # Interpolate flux from the source image using the calculated weights
                    lensed_image[i, j] = source_image[n_y, n_x] * weight_upper_left
                    if n_x + 1 < self._nx_source:
                        lensed_image[i, j] += (
                            source_image[n_y, n_x + 1] * weight_upper_right
                        )
                    if n_y + 1 < self._ny_source:
                        lensed_image[i, j] += (
                            source_image[n_y + 1, n_x] * weight_lower_left
                        )
                        if n_x + 1 < self._nx_source:
                            lensed_image[i, j] += (
                                source_image[n_y + 1, n_x + 1] * weight_lower_right
                            )

        # Apply the ratio (data image pixel area / source grid pixel area) to ensure the flux conservation
        lensed_image *= self._ratio_data_pixel_source_pixel

        return lensed_image

    def sparse_to_array(self, sparse):
        """Converts a sparse image representation (list of `[y_coord, x_coord, value]`
        tuples) to a 2D NumPy array.

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
        """Computes the element-wise sum of products between a sparse matrix and a dense
        2D NumPy array image.

        :param sparse: Sparse matrix representation (list of `[y_coord, x_coord, value]`
            tuples).
        :param ordinary: A 2D NumPy array (dense matrix).
        :returns: The sum of the element-wise products.
        :rtype: float
        """
        sum_temp = 0
        num_element = len(sparse)
        for i in range(num_element):
            sum_temp += sparse[i][2] * ordinary[sparse[i][0], sparse[i][1]]
        return sum_temp

    def sparse_convolve_and_dot_product(self, sp1, sp2, kernel=None):
        """Computes the convolution product of two sparse matrices using a given kernel.
        Equivalent to `(sp1 * kernel) . (sp2)` where '*' is convolution and '.' is dot
        product.

        :param sp1: First sparse matrix representation.
        :param sp2: Second sparse matrix representation.
        :param kernel: The 2D PSF kernel (NumPy array). Assumed to be square with odd dimensions,
                       with its center at the central pixel. If None, `self._kernel` is used
        :returns: The result of the convolution product.
        :rtype: float
        """
        if kernel is None:
            kernel = self._kernel
        inner_product = 0
        num_elements_1 = len(sp1)
        num_elements_2 = len(sp2)
        kernel_center = int(
            len(kernel) / 2
        )  # Assumes kernel is square and has odd dimensions

        for i in range(num_elements_1):
            for j in range(num_elements_2):
                delta_y = sp2[j][0] - sp1[i][0]
                delta_x = sp2[j][1] - sp1[i][1]
                if (
                    kernel_center + delta_y >= 0
                    and kernel_center + delta_y < self._shape_kernel[0]
                    and kernel_center + delta_x >= 0
                    and kernel_center + delta_x < self._shape_kernel[1]
                ):
                    inner_product += (
                        sp1[i][2]
                        * sp2[j][2]
                        * kernel[kernel_center + delta_y, kernel_center + delta_x]
                    )
        return inner_product

    def sparse_convolution(self, sp, kernel=None):
        """Performs convolution of a sparse matrix with a given kernel.

        :param sp: Sparse matrix representation.
        :param kernel: The 2D PSF kernel (NumPy array). Assumed to be square with odd dimensions,
                       with its center at the central pixel. If None, `self._kernel` is used
        :returns: A 2D NumPy array representing the convolved image.
        :rtype: numpy.ndarray
        """
        if kernel is None:
            kernel = self._kernel
        kernel_center = int(
            len(kernel) / 2
        )  # Assumes kernel is square and has odd dimensions
        convolved = np.zeros((self._numPix, self._numPix))
        num_element_sparse = len(sp)

        for i in range(num_element_sparse):
            y_sp, x_sp, val_sp = sp[i][0], sp[i][1], sp[i][2]

            # Calculate slice indices for the kernel relative to the sparse element
            slice_y_start = np.max([kernel_center - y_sp, 0])
            slice_y_end = np.min(
                [kernel_center - y_sp + self._numPix, self._shape_kernel[0]]
            )
            slice_x_start = np.max([kernel_center - x_sp, 0])
            slice_x_end = np.min(
                [kernel_center - x_sp + self._numPix, self._shape_kernel[1]]
            )

            convolved_image_y_start = y_sp - np.min([y_sp, kernel_center])
            convolved_image_y_end = y_sp + np.min(
                [self._numPix - y_sp, self._shape_kernel[0] - kernel_center]
            )
            convolved_image_x_start = x_sp - np.min([x_sp, kernel_center])
            convolved_image_x_end = x_sp + np.min(
                [self._numPix - x_sp, self._shape_kernel[1] - kernel_center]
            )

            convolved[
                convolved_image_y_start:convolved_image_y_end,
                convolved_image_x_start:convolved_image_x_end,
            ] += (
                val_sp * kernel[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
            )
        return convolved

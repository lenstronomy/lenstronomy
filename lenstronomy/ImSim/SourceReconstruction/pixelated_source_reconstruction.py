import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
from tqdm import tqdm

import lenstronomy.Util.util as util

__all__ = ["PixelatedSourceReconstruction"]


class PixelatedSourceReconstruction(object):
    """This class provides methods for pixelated source-plane reconstruction in
    gravitational lensing. It is initialized with data, PSF, lens model, and source
    pixel grid class instances. It provides methods for generating the :math:`M` matrix
    and :math:`b` vector using diagonal image noise covariance matrix specified by C_D,
    or using the interferometric image-plane natwt covariance matrix. Their definitions
    follow arXiv:2508.08393; see the documentation of ``generate_M_b`` for details.

    We use a nested list ``lensed_sp`` to store the contribution from source pixels to
    lensed pixels through strong lensing and bilinear interpolation in the source plane.

    The length of ``lensed_sp`` is equal to the number of source pixels in the reconstruction.
    And each element is
    ``lensed_sp[idx_s] = [[idx_y_lensed, idx_x_lensed, ratio_contributing], ...]``
    where ``idx_s``, ``idx_y_lensed``, ``idx_x_lensed`` are integers.

    Here, ``idx_s`` is the flattened (1D) index of a source pixel.
    Each entry ``[idx_y_lensed, idx_x_lensed, ratio_contributing]`` is a lensed-image pixel that
    receives a contribution from this source pixel, where ``idx_y_lensed`` and ``idx_x_lensed`` are
    its 2D image-plane indices and ``ratio_contributing`` is the corresponding contribution weight.

    For example, if the source pixel has a value of 1.0, it contributes ``1.0 x ratio_contributing`` to the
    specified lensed pixel. A lensed pixel may receive contributions from multiple neighboring source pixels.

    For the interferometric likelihood, ``ratio_contributing`` includes the effect of the primary beam,
    i.e., ratio_contributing = primary beam x lensing effect, if a primary beam is provided.

    We also refer to each element of lensed_sp, i.e., the list ``[[idx_y_lensed, idx_x_lensed, value], ...]``,
    as ``a sparse image`` in the documentation below. It represents an image in which the pixel
    at [idx_y_lensed, idx_x_lensed] has the corresponding ``value``.
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

        self._num_pix = data_class.num_pixel_axes[0]
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
                if self._shape_kernel[check_dim] < 2 * self._num_pix - 1:
                    raise ValueError(
                        "PSF kernel size must be at least (2 * num_pix - 1) "
                        "in each dimension for interferometry_natwt likelihood."
                    )

    def generate_M_b(self, kwargs_lens, verbose=False, show_progress=True):
        """Generates the M matrix and the b vector for source reconstruction based on
        the selected likelihood method.

        :math:`M` and :math:`b` are intermediate quantities used to maximize the likelihood over the source-pixel amplitudes.

        For a pixelated source model, the source image is a linear combination of single-pixel basis images:

        .. math::

            s = \\sum_{i=1}^{N} a_i s_i,

        where :math:`s_i` is the source image containing only the :math:`i`-th source pixel, :math:`a_i` is its amplitude,
        and :math:`N` is the total number of source pixels.
        The chi-square is

        .. math::

            \\chi^2 = (d - BL\\sum_{i=1}^{N}a_i s_i)^T C^{-1}(d - BL\\sum_{j=1}^{N}a_j s_j),

        where :math:`d` is the data, :math:`B` is a data-related linear operator such as PSF convolution,
        :math:`L` is the lensing operator, and :math:`C` is the noise covariance matrix.
        The source amplitudes :math:`a_i` that minimize :math:`\\chi^2` are obtained by solving

        .. math::

            Ma = b,

        where

        .. math::

            M_{ij} = (BLs_i)^T C^{-1}(BLs_j)

        and

        .. math::

            b_i = d^T C^{-1}(BLs_i).

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
        """Generates M and b matrices assuming spatially uncorrelated noise with noise
        covariance specified by data_class.C_D.

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
        lensing_matrix = self._lens_pixel_source_of_a_rectangular_region_csc_matrix(
            kwargs_lens
        )
        if verbose:
            print("Step 1: Finished!")

        if verbose:
            print("Step 2: Convolve the lensed pixels")
        N_lensed = lensing_matrix.shape[1]
        lensed_pixel_conv_set = np.zeros((N_lensed, self._num_pix, self._num_pix))
        for i in tqdm(
            range(N_lensed),
            desc="Running (Convolving lensed pixels)",
            disable=not show_progress,
        ):
            start = lensing_matrix.indptr[i]
            end = lensing_matrix.indptr[i + 1]
            lensed_pixel_conv_set[i] = self._sparse_convolution_from_image_indices(
                lensing_matrix.indices[start:end],
                lensing_matrix.data[start:end],
                self._kernel,
            )
        if verbose:
            print("Step 2: Finished!")

        if verbose:
            print("Step 3: Compute the matrix M and vector b")
        lensed_pixel_conv_set = lensed_pixel_conv_set.reshape(N_lensed, -1)
        image_data = self._image_data.ravel()
        inverse_variance = 1.0 / self._C_D.ravel()

        # b_i = (Ls_i)^T(d/sigma_map)
        b = np.matmul(lensed_pixel_conv_set, (image_data * inverse_variance))

        # M_ij = (Ls_i/sqrt(sigma_map))^T(Ls_j/sqrt(sigma_map))
        lensed_pixel_conv_set *= np.sqrt(inverse_variance)
        M = np.matmul(lensed_pixel_conv_set, lensed_pixel_conv_set.T)

        # Enforce exact symmetry to remove small floating-point asymmetries from matrix multiplication.
        for i in range(1, N_lensed):
            M[i, :i] = M[:i, i]
        if verbose:
            print("Step 3: Finished!")

        return M, b

    def generate_M_b_interferometry_natwt_likelihood(
        self, kwargs_lens, verbose=False, show_progress=True
    ):
        """Generates the M and b matrices for interferometric data with natural
        weighting.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param verbose: If True, print progress messages during matrix generation steps.
            Defaults to False.
        :param show_progress: If True, show progress bar of generating the M matrix.
            Defaults to True.
        :returns: (M, b) tuple, where M is the matrix and b is the vector.
        """
        if verbose:
            print("Step 1: Lensing the source pixels")
        lensing_matrix = self._lens_pixel_source_of_a_rectangular_region_csc_matrix(
            kwargs_lens
        )
        if verbose:
            print("Step 1: Finished!")

        if verbose:
            print(
                "Step 2: Compute the matrix M and vector b (including the convolution step)"
            )
        N_lensed = lensing_matrix.shape[1]
        lensing_matrix_transpose = lensing_matrix.T.tocsr(copy=False)
        M = np.zeros((N_lensed, N_lensed))
        b = lensing_matrix_transpose @ self._image_data.ravel()
        for i in tqdm(
            range(N_lensed),
            desc="Running (iteration times vary)",
            disable=not show_progress,
        ):
            start = lensing_matrix.indptr[i]
            end = lensing_matrix.indptr[i + 1]
            pixel_lensed_convolved = self._sparse_convolution_from_image_indices(
                lensing_matrix.indices[start:end],
                lensing_matrix.data[start:end],
                self._kernel,
            ).ravel()
            products = lensing_matrix_transpose @ pixel_lensed_convolved
            M[i, i:] = products[i:]

        # Exploit symmetry
        for i in range(1, N_lensed):
            M[i, :i] = M[:i, i]
        b /= self._noise_rms**2
        M /= self._noise_rms**2
        if verbose:
            print("Step 2: Finished!")

        return M, b

    def _lens_pixel_source_of_a_rectangular_region_csc_matrix(self, kwargs_lens):
        """Maps image plane pixels to source plane pixels within a specified rectangular
        source grid, considering lensing deflections and applying bilinear
        interpolation.

        This method computes the contribution of each source pixel to each lensed image pixel.
        These contributions form a matrix with shape ``(num_lensed_image_pixels, num_source_pixels)``.
        Because most of its elements are zero, the matrix is stored as a :class:`scipy.sparse.csc_matrix`.
        This function is called directly by the ``generate_M_b`` methods to compute :math:`M` and :math:`b`.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :returns: A CSC sparse matrix with shape = (num_lensed_image_pixels, num_source_pixels).
        :rtype: scipy.sparse.csc_matrix
        """
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

        image_indices = np.arange(self._num_pix**2)
        image_index_parts = []
        source_index_parts = []
        weight_parts = []

        for source_x, source_y, weights in (
            (x_floor, y_floor, w00),
            (x_ceiling, y_floor, w10),
            (x_floor, y_ceiling, w01),
            (x_ceiling, y_ceiling, w11),
        ):
            source_x = source_x.ravel()
            source_y = source_y.ravel()
            weights = weights.ravel()
            valid = (
                (source_x >= 0)
                & (source_x < self._nx_source)
                & (source_y >= 0)
                & (source_y < self._ny_source)
            )
            image_index_parts.append(image_indices[valid])
            source_index_parts.append(
                source_y[valid] * self._nx_source + source_x[valid]
            )
            weight_parts.append(weights[valid])

        image_indices = np.concatenate(image_index_parts)
        source_indices = np.concatenate(source_index_parts)
        weights = np.concatenate(weight_parts)

        # Construct a CSC matrix from image_indices, source_indices, and weights, with data = weights.
        # The CSC matrix has num_lensed_image_pixels rows, so its indices correspond to image_indices.
        # The CSC matrix has num_source_pixels columns, so indptr[1:] is the cumulative sum of elements of source_indices.
        # image_indices, source_indices, and weights should first be sorted in increasing order of source_indices.

        order = np.lexsort((image_indices, source_indices))
        image_indices = image_indices[order]
        source_indices = source_indices[order]
        weights = weights[order]

        indptr = np.empty(self._num_pixel_source + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(
            np.bincount(source_indices, minlength=self._num_pixel_source),
            out=indptr[1:],
        )

        return csc_matrix(
            (weights, image_indices, indptr),
            shape=(self._num_pix**2, self._num_pixel_source),
            copy=False,
        )

    def lens_pixel_source_of_a_rectangular_region(self, kwargs_lens):
        """Maps image plane pixels to source plane pixels within a specified rectangular
        source grid.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :returns: A nested list. Each element in the outer list corresponds to a single source pixel
            within the defined source grid (ordered by their flatten 1D index). Each inner list contains
            elements `[idx_y_lensed, idx_x_lensed, ratio_contributing]`, indicating that the source pixel at this index
            contributes with `ratio_contributing` to the image plane pixel at `[idx_y_lensed, idx_x_lensed]` when lensed.
        :rtype: list
        """
        lensing_matrix = self._lens_pixel_source_of_a_rectangular_region_csc_matrix(
            kwargs_lens
        )
        return self._csc_matrix_to_lensed_sp(lensing_matrix)

    def _csc_matrix_to_lensed_sp(self, csc):
        """Converts a CSC lensing matrix to the lensed_sp.

        :param csc: The CSC matrix with shape = (num_lensed_image_pixels,
            num_source_pixels).
        :returns: A nested list containing the image pixel indices and contributions
            from each source pixel.
        :rtype: list
        """

        if not isspmatrix_csc(csc):
            raise TypeError("csc must be a scipy.sparse.csc_matrix.")

        num_lensed_pixels, num_source_pixels = csc.shape
        if num_lensed_pixels != self._num_pix**2:
            raise ValueError(
                "The number of rows in the CSC matrix should be equal to the number of lensed image pixels."
            )
        if num_source_pixels != self._num_pixel_source:
            raise ValueError(
                "The number of columns in the CSC matrix should be equal to the number of source pixels."
            )

        if not csc.has_sorted_indices:
            csc = csc.copy()
            csc.sort_indices()

        image_y, image_x = np.divmod(csc.indices, self._num_pix)
        return [
            [
                [int(image_y[index]), int(image_x[index]), csc.data[index]]
                for index in range(csc.indptr[source], csc.indptr[source + 1])
            ]
            for source in range(csc.shape[1])
        ]

    def lens_an_image_by_rayshooting(self, kwargs_lens, source_image):
        """Lenses a pixelated source image to the image plane using ray-shooting and
        bilinear interpolation. The input image should have the same dimension and
        coordinates defined by source_pixel_grid_class.

        This method works by ray-shooting image plane pixels back to the source plane to
        find the corresponding source coordinate, and then interpolating the flux from the input
        source image at that coordinate.

        Note that the primary beam will NOT be applied on the lensed image for interferometric data.

        :param kwargs_lens: List of keyword arguments for the lens_model_class.
        :param image: 2D NumPy array representing the pixelated source plane image. Expected to have
                      dimensions defined by source_pixel_grid_class.
        :type image: numpy.ndarray
        :returns: 2D NumPy array representing the lensed image in the image plane.
        :rtype: numpy.ndarray
        :raises ValueError: If the input `source_image` dimensions do not match the dimensions of the
                            defined source pixel grid (`self._ny_source`, `self._nx_source`).
        """

        ny_source_check, nx_source_check = np.shape(source_image)
        if nx_source_check != self._nx_source or ny_source_check != self._ny_source:
            raise ValueError(
                f"Input image size ({ny_source_check}, {nx_source_check}) must match the defined "
                f"source grid class dimension ({self._ny_source}, {self._nx_source})."
            )

        beta_x_grid_2d, beta_y_grid_2d = self._lens_model_class.ray_shooting(
            self._x_grid_data, self._y_grid_data, kwargs=kwargs_lens
        )
        beta_x_grid = util.image2array(beta_x_grid_2d)
        beta_y_grid = util.image2array(beta_y_grid_2d)

        # Compute the floor indices of the ray-shot image pixels in the source plane.
        n_x = np.floor(
            (beta_x_grid - self._source_min_x) / self._pixel_width_source
        ).astype(int)
        n_y = np.floor(
            (beta_y_grid - self._source_min_y) / self._pixel_width_source
        ).astype(int)

        # If the ray shoots the image pixel outside the defined source image boundaries, valid = False
        # valid = True means the image pixels are rayshot back to the source plane within the source image region
        valid = (
            (n_x >= 0) & (n_x < self._nx_source) & (n_y >= 0) & (n_y < self._ny_source)
        )

        lensed_image = np.zeros(self._num_pix**2)
        n_x_valid = n_x[valid]
        n_y_valid = n_y[valid]
        beta_x_valid = beta_x_grid[valid]
        beta_y_valid = beta_y_grid[valid]

        # Calculate bilinear interpolation weights for the four surrounding source pixels, for valid image pixels
        # These weights depend on the sub-pixel position of (beta_y_valid, beta_x_valid)
        # within the source pixel (n_y_valid, n_x_valid).

        weight_upper_left = (
            np.abs(
                self._source_min_y
                + n_y_valid * self._pixel_width_source
                + self._pixel_width_source
                - beta_y_valid
            )
            * np.abs(
                self._source_min_x
                + n_x_valid * self._pixel_width_source
                + self._pixel_width_source
                - beta_x_valid
            )
            / (self._pixel_width_source**2)
        )
        weight_upper_right = (
            np.abs(
                self._source_min_y
                + n_y_valid * self._pixel_width_source
                + self._pixel_width_source
                - beta_y_valid
            )
            * np.abs(
                self._source_min_x + n_x_valid * self._pixel_width_source - beta_x_valid
            )
            / (self._pixel_width_source**2)
        )
        weight_lower_left = (
            np.abs(
                self._source_min_x
                + n_x_valid * self._pixel_width_source
                + self._pixel_width_source
                - beta_x_valid
            )
            * np.abs(
                self._source_min_y + n_y_valid * self._pixel_width_source - beta_y_valid
            )
            / (self._pixel_width_source**2)
        )
        weight_lower_right = (
            np.abs(
                self._source_min_x + n_x_valid * self._pixel_width_source - beta_x_valid
            )
            * np.abs(
                self._source_min_y + n_y_valid * self._pixel_width_source - beta_y_valid
            )
            / (self._pixel_width_source**2)
        )

        # Interpolate flux from the source image using the calculated weights for valid image pixels
        valid_values = source_image[n_y_valid, n_x_valid] * weight_upper_left

        # For all valid image pixels
        # "valid_upper_right = True" means the right neighbor source pixel is also in the source region
        valid_upper_right = n_x_valid + 1 < self._nx_source
        valid_values[valid_upper_right] += (
            source_image[n_y_valid[valid_upper_right], n_x_valid[valid_upper_right] + 1]
            * weight_upper_right[valid_upper_right]
        )

        valid_lower_left = n_y_valid + 1 < self._ny_source
        valid_values[valid_lower_left] += (
            source_image[n_y_valid[valid_lower_left] + 1, n_x_valid[valid_lower_left]]
            * weight_lower_left[valid_lower_left]
        )

        valid_lower_right = valid_upper_right & valid_lower_left
        valid_values[valid_lower_right] += (
            source_image[
                n_y_valid[valid_lower_right] + 1,
                n_x_valid[valid_lower_right] + 1,
            ]
            * weight_lower_right[valid_lower_right]
        )

        lensed_image[valid] = valid_values
        lensed_image = lensed_image.reshape(self._num_pix, self._num_pix)

        # Apply the ratio (data image pixel area / source grid pixel area) to ensure the flux conservation
        lensed_image *= self._ratio_data_pixel_source_pixel

        return lensed_image

    def sparse_to_array(self, sparse):
        """Converts a sparse image representation (list of `[idx_y, idx_x, value]`) to a
        2D NumPy array.

        :param sparse: A list representing non-zero elements of the sparse image.
        :returns: A 2D NumPy array representing the full image.
        :rtype: numpy.ndarray
        """
        image = np.zeros((self._num_pix, self._num_pix))
        num_of_elements = len(sparse)
        for i in range(num_of_elements):
            image[sparse[i][0], sparse[i][1]] = sparse[i][2]
        return image

    @staticmethod
    def sum_sparse_elementwise_product(sparse, ordinary):
        """Computes the element-wise sum of products between a sparse image and a dense
        2D NumPy array image.

        :param sparse: A sparse image representation (list of `[idx_y, idx_x, value]`).
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
        """Computes the convolution product of two sparse images using a given kernel.
        Equivalent to `(sp1 * kernel) . (sp2)` where `*` is convolution and `.` is dot
        product.

        :param sp1: First sparse image.
        :param sp2: Second sparse image.
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
        """Performs convolution of a sparse image with a given kernel.

        :param sp: A sparse image.
        :param kernel: The 2D PSF kernel (NumPy array). Assumed to be square with odd dimensions,
                       with its center at the central pixel. If None, `self._kernel` is used
        :returns: A 2D NumPy array representing the convolved image.
        :rtype: numpy.ndarray
        """
        if kernel is None:
            kernel = self._kernel

        if len(sp) == 0:
            return np.zeros((self._num_pix, self._num_pix))

        sparse_array = np.asarray(sp)
        indices = sparse_array[:, 0].astype(np.intp) * self._num_pix + sparse_array[
            :, 1
        ].astype(np.intp)
        return self._sparse_convolution_from_image_indices(
            indices, sparse_array[:, 2], kernel
        )

    def _sparse_convolution_from_image_indices(self, indices, values, kernel=None):
        """Performs convolution from flattened sparse indices and values.

        :param indices: Flattened image pixel indices.
        :param values: Values at the corresponding image pixel indices.
        :param kernel: The 2D PSF kernel. If None, `self._kernel` is used.
        :returns: A 2D NumPy array representing the convolved image.
        :rtype: numpy.ndarray
        """
        if kernel is None:
            kernel = self._kernel

        # Assume kernel is square and has odd dimensions
        kernel_center = kernel.shape[0] // 2
        convolved = np.zeros((self._num_pix, self._num_pix))

        for index, val_sp in zip(indices, values):
            y_sp = index // self._num_pix
            x_sp = index % self._num_pix

            # Calculate slice indices for the kernel relative to the sparse element
            slice_y_start = max(kernel_center - y_sp, 0)
            slice_y_end = min(kernel_center - y_sp + self._num_pix, kernel.shape[0])
            slice_x_start = max(kernel_center - x_sp, 0)
            slice_x_end = min(kernel_center - x_sp + self._num_pix, kernel.shape[1])

            convolved_image_y_start = y_sp - min(y_sp, kernel_center)
            convolved_image_y_end = y_sp + min(
                self._num_pix - y_sp, kernel.shape[0] - kernel_center
            )
            convolved_image_x_start = x_sp - min(x_sp, kernel_center)
            convolved_image_x_end = x_sp + min(
                self._num_pix - x_sp, kernel.shape[1] - kernel_center
            )

            convolved[
                convolved_image_y_start:convolved_image_y_end,
                convolved_image_x_start:convolved_image_x_end,
            ] += (
                val_sp * kernel[slice_y_start:slice_y_end, slice_x_start:slice_x_end]
            )
        return convolved

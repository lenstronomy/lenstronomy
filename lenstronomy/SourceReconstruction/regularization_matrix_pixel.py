"""This module provides functions to construct regularization matrices for pixelated
source plane reconstructions in gravitational lensing. These matrices are used to
constrain the pixel amplitudes in linear inversion problems, promoting desired
properties such as smoothness.

Currently supported regularization types include:
- Zeroth-order (L2 norm on pixel amplitudes)
- Gradient (L2 norm on spatial gradients, promoting smoothness)
- Curvature (L2 norm on second-order derivatives, promoting smoother variations)
"""

import numpy as np

__all__ = ["pixelated_regularization_matrix"]


def pixelated_regularization_matrix(xlen, ylen, regularization_type):
    """Constructs the regularization matrix for a rectangular pixelated source region.

    The regularization term for pixel amplitudes :math:`\\mathbf{a}` (flattened into a 1D vector)
    is generally expressed as :math:`\\mathbf{a}^T U \\mathbf{a}`, where :math:`U` is the
    regularization matrix. This function returns the matrix :math:`U`.

    The pixels are aligned in a 1D vector `a` by flattening the 2D source grid in a row-major
    order. Specifically, if a pixel has 2D indices :math:`(i_x, i_y)` (where :math:`i_x` is
    the column index from 0 to `xlen-1` and :math:`i_y` is the row index from 0 to `ylen-1`),
    its 1D index :math:`i` is given by :math:`i = i_y \\cdot xlen + i_x`.

    :param xlen: int, Number of pixels in the x-direction (horizontal dimension of the source grid).
    :param ylen: int, Number of pixels in the y-direction (vertical dimension of the source grid).
    :param regularization_type: str, Type of regularization to apply.
                                Supported options are 'zeroth_order', 'gradient', 'curvature'.
    :return: numpy.ndarray, The regularization matrix :math:`U` with shape `(xlen * ylen, xlen * ylen)`.
    :raises TypeError: If `xlen` or `ylen` are not integers.
    :raises ValueError: If an unsupported `regularization_type` is provided.
    """
    if not isinstance(xlen, int):
        raise TypeError(f"xlen must be an integer, but got {type(xlen).__name__}.")
    if not isinstance(ylen, int):
        raise TypeError(f"ylen must be an integer, but got {type(ylen).__name__}.")
    if regularization_type == "zeroth_order":
        return _zeroth_order_regularization_matrix_pixel(xlen, ylen)
    elif regularization_type == "gradient":
        return _gradient_regularization_matrix_pixel(xlen, ylen)
    elif regularization_type == "curvature":
        return _curvature_regularization_matrix_pixel(xlen, ylen)
    else:
        raise ValueError(
            f"Unsupported regularization_type: '{regularization_type}'. "
            "Supported options are: 'zeroth_order', 'gradient', 'curvature'."
        )


def _zeroth_order_regularization_matrix_pixel(xlen, ylen):
    """Constructs the zeroth-order regularization matrix.

    The zeroth-order regularization term penalizes the amplitude of each pixel,
    defined as the sum of squared amplitudes:

    .. math::
        \\sum_{i} (a_i)^2

    This corresponds to a diagonal regularization matrix where all diagonal elements are 1.

    :param xlen: int, Number of pixels in the x-direction.
    :param ylen: int, Number of pixels in the y-direction.
    :return: numpy.ndarray, The (xlen * ylen, xlen * ylen) sized identity matrix.
    """
    Umatrix = np.identity(xlen * ylen)
    return Umatrix


def _gradient_regularization_matrix_pixel(xlen, ylen):
    """Constructs the gradient (first-order) regularization matrix.

    The gradient regularization term penalizes large differences between adjacent pixels,
    promoting a smoother reconstructed source. It is defined as the sum of squared
    differences between horizontally and vertically adjacent pixel amplitudes:

    .. math::
        \\sum_{ij}U_{ij} a_i a_j = \\sum_{i_x=0}^{n}\\sum_{i_y=0}^{m} ((a_{i_x,i_y} - a_{i_x+1,i_y})^2 +
        (a_{i_x,i_y} - a_{i_x,i_y+1})^2\),

    where the summation is over all pixels.
    Pixels outside the defined rectangular region (`xlen`, `ylen`) are implicitly
    treated as having zero amplitude (Dirichlet boundary conditions).

    :return: numpy.ndarray, The (xlen * ylen, xlen * ylen) sized gradient regularization matrix.
    """
    num_pixels_total = xlen * ylen
    Umatrix = np.zeros((num_pixels_total, num_pixels_total))

    block0 = 4 * np.identity(xlen)
    for i in range(xlen - 1):
        block0[i, i + 1] = -1
        block0[i + 1, i] = -1

    block1 = -1 * np.identity(xlen)

    # Populate the main diagonal blocks with block0
    for i_y in range(ylen):
        Umatrix[xlen * i_y : xlen * (i_y + 1), xlen * i_y : xlen * (i_y + 1)] = block0

    # Populate the off-diagonal blocks with block1
    for i_y in range(ylen - 1):
        Umatrix[xlen * (i_y + 1) : xlen * (i_y + 2), xlen * i_y : xlen * (i_y + 1)] = (
            block1
        )
        Umatrix[xlen * i_y : xlen * (i_y + 1), xlen * (i_y + 1) : xlen * (i_y + 2)] = (
            block1
        )

    return Umatrix


def _curvature_regularization_matrix_pixel(xlen, ylen):
    """Constructs the curvature (second-order) regularization matrix.

    The curvature regularization term penalizes variations in the gradient, promoting
    even smoother reconstructions. It is defined as the sum of squared discrete second
    derivatives (Laplacian-like operators) in x and y directions:

    .. math::
        \\sum_{ij}U_{ij} a_i a_j = \\sum_{i_x=0}^{n+1}\\sum_{i_y=0}^{m+1} ((2 a_{i_x,i_y} - a_{i_x+1,i_y} - a_{i_x-1,i_y})^2 +
        (2 a_{i_x,i_y} - a_{i_x,i_y+1} - a_{i_x,i_y-1})^2),

    Pixels outside the defined rectangular region (`xlen`, `ylen`) are implicitly
    treated as having zero amplitude (Dirichlet boundary conditions).

    :return: numpy.ndarray, The (xlen * ylen, xlen * ylen) sized curvature regularization matrix.
    """
    num_pixels_total = xlen * ylen
    Umatrix = np.zeros((num_pixels_total, num_pixels_total))

    block0 = np.zeros((xlen, xlen))
    for i in range(xlen):
        block0[i, i] = 12
    for i in range(xlen - 1):
        block0[i, i + 1] = -4
        block0[i + 1, i] = -4
    for i in range(xlen - 2):
        block0[i, i + 2] = 1
        block0[i + 2, i] = 1

    block1 = -4 * np.identity(xlen)
    block2 = np.identity(xlen)

    for i_y in range(ylen):
        Umatrix[xlen * i_y : xlen * (i_y + 1), xlen * i_y : xlen * (i_y + 1)] = block0

    for i_y in range(ylen - 1):
        Umatrix[xlen * (i_y + 1) : xlen * (i_y + 2), xlen * i_y : xlen * (i_y + 1)] = (
            block1
        )
        Umatrix[xlen * i_y : xlen * (i_y + 1), xlen * (i_y + 1) : xlen * (i_y + 2)] = (
            block1
        )

    for i_y in range(ylen - 2):
        Umatrix[xlen * (i_y + 2) : xlen * (i_y + 3), xlen * i_y : xlen * (i_y + 1)] = (
            block2
        )
        Umatrix[xlen * i_y : xlen * (i_y + 1), xlen * (i_y + 2) : xlen * (i_y + 3)] = (
            block2
        )

    return Umatrix

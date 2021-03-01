__author__ = 'Simon Birrer'

"""
this file contains standard routines
"""

import numpy as np
import mpmath
import itertools
from lenstronomy.Util.numba_util import jit
from lenstronomy.Util.package_util import exporter

export, __all__ = exporter()


@export
def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


@export
def approx_theta_E(ximg, yimg):
    dis = []
    xinds, yinds = [0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]

    for (i, j) in zip(xinds, yinds):
        dx, dy = ximg[i] - ximg[j], yimg[i] - yimg[j]
        dr = (dx ** 2 + dy ** 2) ** 0.5
        dis.append(dr)
    dis = np.array(dis)

    greatest = np.argmax(dis)
    dr_greatest = dis[greatest]
    dis[greatest] = 0

    second_greatest = np.argmax(dis)
    dr_second = dis[second_greatest]

    return 0.5 * (dr_greatest * dr_second) ** 0.5


@export
def sort_image_index(ximg, yimg, xref, yref):
    """

    :param ximg: x coordinates to sort
    :param yimg: y coordinates to sort
    :param xref: reference x coordinate
    :param yref: reference y coordinate
    :return: indexes such that ximg[indexes],yimg[indexes] matches xref,yref
    """

    assert len(xref) == len(ximg)
    ximg, yimg = np.array(ximg), np.array(yimg)
    x_self = np.array(list(itertools.permutations(ximg)))
    y_self = np.array(list(itertools.permutations(yimg)))

    indexes = [0, 1, 2, 3]
    index_iterations = list(itertools.permutations(indexes))
    delta_r = []

    for i in range(0, int(len(x_self))):
        dr = 0
        for j in range(0, int(len(x_self[0]))):
            dr += (x_self[i][j] - xref[j]) ** 2 + (y_self[i][j] - yref[j]) ** 2

        delta_r.append(dr ** .5)

    min_indexes = np.array(index_iterations[np.argmin(delta_r)])

    return min_indexes


@export
@jit()
def rotate(xcoords, ycoords, angle):
    """

    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    return xcoords * np.cos(angle) + ycoords * np.sin(angle), -xcoords * np.sin(angle) + ycoords * np.cos(angle)


@export
def map_coord2pix(ra, dec, x_0, y_0, M):
    """
    this routines performs a linear transformation between two coordinate systems. Mainly used to transform angular
    into pixel coordinates in an image
    :param ra: ra coordinates
    :param dec: dec coordinates
    :param x_0: pixel value in x-axis of ra,dec = 0,0
    :param y_0: pixel value in y-axis of ra,dec = 0,0
    :param M: 2x2 matrix to transform angular to pixel coordinates
    :return: transformed coordinate systems of input ra and dec
    """
    x, y = M.dot(np.array([ra, dec]))
    return x + x_0, y + y_0


@export
def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array
    (only works when length of array is n**2, or nx and ny are provided)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(len(array)))
        if n ** 2 != len(array):
            raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(array)))
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image


@export
def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx * ny)  # change the shape to be 1d
    return imgh


@export
def array2cube(array, n_1, n_23):
    """
    returns the information contained in a 1d array of shape (n_1*n_23*n_23) into 3d array with shape (n_1, sqrt(n_23), sqrt(n_23))

    :param array: image values
    :type array: 1d array
    :param n_1: first dimension of returned array
    :type int:
    :param n_23: square of second and third dimensions of returned array
    :type int:
    :returns: 3d array
    :raises ValueError: when n_23 is not a perfect square
    """
    n = int(np.sqrt(n_23))
    if n ** 2 != n_23:
        raise ValueError("2nd and 3rd dims (%s) are not square of integer number!" % n_23)
    n_2, n_3 = n, n
    cube = array.reshape(n_1, n_2, n_3)
    return cube


@export
def cube2array(cube):
    """
    returns the information contained in a 3d array of shape (n_1, n_2, n_3) into 1d array with shape (n_1*n_2*n_3)

    :param array: image values
    :type array: 3d array
    :returns: 1d array
    """
    n_1, n_2, n_3 = cube.shape
    array = cube.reshape(n_1 * n_2 * n_3)
    return array


@export
def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """
    creates pixel grid (in 1d arrays of x- and y- positions)
    default coordinate frame is such that (0,0) is in the center of the coordinate grid

    :param numPix: number of pixels per axis
        Give an integers for a square grid, or a 2-length sequence
        (first, second axis length) for a non-square grid.
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    # Check numPix is an integer, or 2-sequence of integers
    if isinstance(numPix, (tuple, list, np.ndarray)):
        assert len(numPix) == 2
        if any(x != round(x) for x in numPix):
            raise ValueError("numPix contains non-integers: %s" % numPix)
        numPix = np.asarray(numPix, dtype=np.int)
    else:
        if numPix != round(numPix):
            raise ValueError("Attempt to specify non-int numPix: %s" % numPix)
        numPix = np.array([numPix, numPix], dtype=np.int)

    # Super-resolution sampling
    numPix_eff = (numPix * subgrid_res).astype(np.int)
    deltapix_eff = deltapix / float(subgrid_res)

    # Compute unshifted grids.
    # X values change quickly, Y values are repeated many times
    x_grid = np.tile(np.arange(numPix_eff[0]), numPix_eff[1]) * deltapix_eff
    y_grid = np.repeat(np.arange(numPix_eff[1]), numPix_eff[0]) * deltapix_eff

    if left_lower is True:
        # Shift so (0, 0) is in the "lower left"
        # Note this does not shift when subgrid_res = 1
        shift = -1. / 2 + 1. / (2 * subgrid_res) * np.array([1, 1])
    else:
        # Shift so (0, 0) is centered
        shift = deltapix_eff * (numPix_eff - 1) / 2

    return x_grid - shift[0], y_grid - shift[1]


@export
def make_grid_transformed(numPix, Mpix2Angle):
    """
    returns grid with linear transformation (deltaPix and rotation)
    :param numPix: number of Pixels
    :param Mpix2Angle: 2-by-2 matrix to mat a pixel to a coordinate
    :return: coordinate grid
    """
    x_grid, y_grid = make_grid(numPix, deltapix=1)
    ra_grid, dec_grid = map_coord2pix(x_grid, y_grid, 0, 0, Mpix2Angle)
    return ra_grid, dec_grid


@export
def make_grid_with_coordtransform(numPix, deltapix, subgrid_res=1, center_ra=0, center_dec=0, left_lower=False,
                                  inverse=True):
    """
    same as make_grid routine, but returns the transformation matrix and shift between coordinates and pixel

    :param numPix: number of pixels per axis
    :param deltapix: pixel scale per axis
    :param subgrid_res: supersampling resolution relative to the stated pixel size
    :param center_ra: center of the grid
    :param center_dec: center of the grid
    :param left_lower: sets the zero point at the lower left corner of the pixels
    :param inverse: bool, if true sets East as left, otherwise East is righrt
    :return:
    """
    numPix_eff = numPix * subgrid_res
    deltapix_eff = deltapix / float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if inverse is True:
        delta_x = -deltapix_eff
    else:
        delta_x = deltapix_eff
    if left_lower is True:
        ra_grid = matrix[:, 0] * delta_x
        dec_grid = matrix[:, 1] * deltapix_eff
    else:
        ra_grid = (matrix[:, 0] - (numPix_eff - 1) / 2.) * delta_x
        dec_grid = (matrix[:, 1] - (numPix_eff - 1) / 2.) * deltapix_eff
    shift = (subgrid_res - 1) / (2. * subgrid_res) * deltapix
    ra_grid -= shift + center_ra
    dec_grid -= shift + center_dec
    ra_at_xy_0 = ra_grid[0]
    dec_at_xy_0 = dec_grid[0]

    Mpix2coord = np.array([[delta_x, 0], [0, deltapix_eff]])
    Mcoord2pix = np.linalg.inv(Mpix2coord)
    x_at_radec_0, y_at_radec_0 = map_coord2pix(-ra_at_xy_0, -dec_at_xy_0, x_0=0, y_0=0, M=Mcoord2pix)
    return ra_grid, dec_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix


@export
def grid_from_coordinate_transform(nx, ny, Mpix2coord, ra_at_xy_0, dec_at_xy_0):
    """
    return a grid in x and y coordinates that satisfy the coordinate system


    :param nx: number of pixels in x-axis
    :param ny: number of pixels in y-axis
    :param Mpix2coord: transformation matrix (2x2) of pixels into coordinate displacements
    :param ra_at_xy_0: RA coordinate at (x,y) = (0,0)
    :param dec_at_xy_0: DEC coordinate at (x,y) = (0,0)
    :return: RA coordinate grid, DEC coordinate grid
    """
    a = np.arange(nx)
    b = np.arange(ny)
    matrix = np.dstack(np.meshgrid(a, b)).reshape(-1, 2)
    x_grid = matrix[:, 0]
    y_grid = matrix[:, 1]
    ra_grid = x_grid * Mpix2coord[0, 0] + y_grid * Mpix2coord[0, 1] + ra_at_xy_0
    dec_grid = x_grid * Mpix2coord[1, 0] + y_grid * Mpix2coord[1, 1] + dec_at_xy_0
    return ra_grid, dec_grid


@export
def get_axes(x, y):
    """
    computes the axis x and y of a given 2d grid
    :param x:
    :param y:
    :return:
    """
    n = int(np.sqrt(len(x)))
    if n ** 2 != len(x):
        raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(x)))
    x_image = x.reshape(n, n)
    y_image = y.reshape(n, n)
    x_axes = x_image[0, :]
    y_axes = y_image[:, 0]
    return x_axes, y_axes


@export
def averaging(grid, numGrid, numPix):
    """
    resize 2d pixel grid with numGrid to numPix and averages over the pixels
    :param grid: higher resolution pixel grid
    :param numGrid: number of pixels per axis in the high resolution input image
    :param numPix: lower number of pixels per axis in the output image (numGrid/numPix is integer number)
    :return:
    """

    Nbig = numGrid
    Nsmall = numPix
    small = grid.reshape([int(Nsmall), int(Nbig / Nsmall), int(Nsmall), int(Nbig / Nsmall)]).mean(3).mean(1)
    return small


@export
def displaceAbs(x, y, sourcePos_x, sourcePos_y):
    """
    calculates a grid of distances to the observer in angel

    :param mapped_cartcoord: mapped cartesian coordinates
    :type mapped_cartcoord: numpy array (n,2)
    :param sourcePos: source position
    :type sourcePos: numpy vector [x0,y0]
    :returns:  array of displacement
    :raises: AttributeError, KeyError
    """
    x_mapped = x - sourcePos_x
    y_mapped = y - sourcePos_y
    absmapped = np.sqrt(x_mapped ** 2 + y_mapped ** 2)
    return absmapped


@export
def get_distance(x_mins, y_mins, x_true, y_true):
    """

    :param x_mins:
    :param y_mins:
    :param x_true:
    :param y_true:
    :return:
    """
    if len(x_mins) != len(x_true):
        return 10 ** 10
    dist = 0
    x_true_list = np.array(x_true)
    y_true_list = np.array(y_true)

    for i in range(0, len(x_mins)):
        dist_list = (x_mins[i] - x_true_list) ** 2 + (y_mins[i] - y_true_list) ** 2
        dist += min(dist_list)
        k = np.where(dist_list == min(dist_list))
        if type(k) != int:
            k = k[0]
        x_true_list = np.delete(x_true_list, k)
        y_true_list = np.delete(y_true_list, k)
    return dist


@export
def compare_distance(x_mapped, y_mapped):
    """

    :param x_mapped: array of x-positions of remapped catalogue image
    :param y_mapped: array of y-positions of remapped catalogue image
    :return: sum of distance square of positions
    """
    X2 = 0
    for i in range(0, len(x_mapped) - 1):
        for j in range(i + 1, len(x_mapped)):
            dx = x_mapped[i] - x_mapped[j]
            dy = y_mapped[i] - y_mapped[j]
            X2 += dx ** 2 + dy ** 2
    return X2


@export
def min_square_dist(x_1, y_1, x_2, y_2):
    """
    return minimum of quadratic distance of pairs (x1, y1) to pairs (x2, y2)
    :param x_1:
    :param y_1:
    :param x_2:
    :param y_2:
    :return:
    """
    dist = np.zeros_like(x_1)
    for i in range(len(x_1)):
        dist[i] = np.min((x_1[i] - x_2) ** 2 + (y_1[i] - y_2) ** 2)
    return dist


@export
def selectBest(array, criteria, numSelect, highest=True):
    """

    :param array: numpy array to be selected from
    :param criteria: criteria of selection
    :param highest: bool, if false the lowest will be selected
    :param numSelect: number of elements to be selected
    :return:
    """
    n = len(array)
    m = len(criteria)
    if n != m:
        raise ValueError('Elements in array (%s) not equal to elements in criteria (%s)' % (n, m))
    if n < numSelect:
        return array
    array_sorted = array[criteria.argsort()]
    if highest:
        result = array_sorted[n - numSelect:]
    else:
        result = array_sorted[0:numSelect]
    return result[::-1]


@export
def select_best(array, criteria, num_select, highest=True):
    """

    :param array: numpy array to be selected from
    :param criteria: criteria of selection
    :param highest: bool, if false the lowest will be selected
    :param num_select: number of elements to be selected
    :return:
    """
    n = len(array)
    m = len(criteria)
    if n != m:
        raise ValueError('Elements in array (%s) not equal to elements in criteria (%s)' % (n, m))
    if n < num_select:
        return array
    array = np.array(array)
    if highest is True:
        indexes = criteria.argsort()[::-1][:num_select]
    else:
        indexes = criteria.argsort()[::-1][n - num_select:]
    return array[indexes]


@export
def points_on_circle(radius, num_points, connect_ends=True):
    """
    returns a set of uniform points around a circle
    :param radius: radius of the circle
    :param num_points: number of points on the circle
    :param connect_ends: boolean, if True, start and end point are the same
    :return: x-coords, y-coords of points on the circle
    """
    if connect_ends:
        angle = np.linspace(0, 2 * np.pi, num_points)
    else:
        angle = np.linspace(0, 2 * np.pi * (1 - 1./num_points), num_points)
    x_coord = np.cos(angle) * radius
    y_coord = np.sin(angle) * radius
    return x_coord, y_coord


@export
@jit()
def neighborSelect(a, x, y):
    """
    #TODO replace by from scipy.signal import argrelextrema for speed up
    >>> from scipy.signal import argrelextrema
    >>> x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
    >>> argrelextrema(x, np.greater)
    (array([3, 6]),)
    >>> y = np.array([[1, 2, 1, 2],
    ...               [2, 2, 0, 0],
    ...               [5, 3, 4, 4]])
    ...
    >>> argrelextrema(y, np.less, axis=1)
    (array([0, 2]), array([2, 1]))


    finds (local) minima in a 2d grid

    :param a: 1d array of displacements from the source positions
    :type a: numpy array with length numPix**2 in float
    :returns:  array of indices of local minima, values of those minima
    :raises: AttributeError, KeyError
    """
    dim = int(np.sqrt(len(a)))
    values = []
    x_mins = []
    y_mins = []
    for i in range(dim + 1, len(a) - dim - 1):
        if (a[i] < a[i - 1]
                and a[i] < a[i + 1]
                and a[i] < a[i - dim]
                and a[i] < a[i + dim]
                and a[i] < a[i - (dim - 1)]
                and a[i] < a[i - (dim + 1)]
                and a[i] < a[i + (dim - 1)]
                and a[i] < a[i + (dim + 1)]):
            if (a[i] < a[(i - 2 * dim - 1) % dim ** 2]
                    and a[i] < a[(i - 2 * dim + 1) % dim ** 2]
                    and a[i] < a[(i - dim - 2) % dim ** 2]
                    and a[i] < a[(i - dim + 2) % dim ** 2]
                    and a[i] < a[(i + dim - 2) % dim ** 2]
                    and a[i] < a[(i + dim + 2) % dim ** 2]
                    and a[i] < a[(i + 2 * dim - 1) % dim ** 2]
                    and a[i] < a[(i + 2 * dim + 1) % dim ** 2]
                    and a[i] < a[(i + 2 * dim) % dim ** 2]
                    and a[i] < a[(i - 2 * dim) % dim ** 2]
                    and a[i] < a[(i - 2) % dim ** 2]
                    and a[i] < a[(i + 2) % dim ** 2]):
                x_mins.append(x[i])
                y_mins.append(y[i])
                values.append(a[i])
    return np.array(x_mins), np.array(y_mins), np.array(values)


@export
def fwhm2sigma(fwhm):
    """

    :param fwhm: full-widt-half-max value
    :return: gaussian sigma (sqrt(var))
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma


@export
def sigma2fwhm(sigma):
    """

    :param sigma:
    :return:
    """
    fwhm = sigma * (2 * np.sqrt(2 * np.log(2)))
    return fwhm


@export
def hyper2F2_array(a, b, c, d, x):
    """

    :param a:
    :param b:
    :param c:
    :param d:
    :param x:
    :return:
    """
    if isinstance(x, int) or isinstance(x, float):
        out = mpmath.hyp2f2(a, b, c, d, x)
    else:
        n = len(x)
        out = np.zeros(n)
        for i in range(n):
            out[i] = mpmath.hyp2f2(a, b, c, d, x[i])
    return out


@export
def make_subgrid(ra_coord, dec_coord, subgrid_res=2):
    """
    return a grid with subgrid resolution
    :param ra_coord:
    :param dec_coord:
    :param subgrid_res:
    :return:
    """
    ra_array = array2image(ra_coord)
    dec_array = array2image(dec_coord)
    n = len(ra_array)
    d_ra_x = ra_array[0][1] - ra_array[0][0]
    d_ra_y = ra_array[1][0] - ra_array[0][0]
    d_dec_x = dec_array[0][1] - dec_array[0][0]
    d_dec_y = dec_array[1][0] - dec_array[0][0]

    ra_array_new = np.zeros((n * subgrid_res, n * subgrid_res))
    dec_array_new = np.zeros((n * subgrid_res, n * subgrid_res))
    for i in range(0, subgrid_res):
        for j in range(0, subgrid_res):
            ra_array_new[i::subgrid_res, j::subgrid_res] = ra_array + d_ra_x * (
                        -1 / 2. + 1 / (2. * subgrid_res) + j / float(subgrid_res)) + d_ra_y * (
                                                                       -1 / 2. + 1 / (2. * subgrid_res) + i / float(
                                                                   subgrid_res))
            dec_array_new[i::subgrid_res, j::subgrid_res] = dec_array + d_dec_x * (
                        -1 / 2. + 1 / (2. * subgrid_res) + j / float(subgrid_res)) + d_dec_y * (
                                                                        -1 / 2. + 1 / (2. * subgrid_res) + i / float(
                                                                    subgrid_res))

    ra_coords_sub = image2array(ra_array_new)
    dec_coords_sub = image2array(dec_array_new)
    return ra_coords_sub, dec_coords_sub


@export
def convert_bool_list(n, k=None):
    """
    returns a bool list of the length of the lens models
    if k = None: returns bool list with True's
    if k is int, returns bool list with False's but k'th is True
    if k is a list of int, e.g. [0, 3, 5], returns a bool list with True's in the integers listed and False elsewhere
    if k is a boolean list, checks for size to match the numbers of models and returns it

    :param n: integer, total lenght of output boolean list
    :param k: None, int, or list of ints
    :return: bool list
    """
    if k is None:
        bool_list = [True] * n
    elif isinstance(k, (int, np.integer)):  # single integer
        bool_list = [False] * n
        bool_list[k] = True
    elif len(k) == 0:  # empty list
        bool_list = [False] * n
    elif isinstance(k[0], bool):
        if n != len(k):
            raise ValueError('length of selected lens models in format of boolean list is %s '
                             'and does not match the models of this class instance %s.' % (len(k), n))
        bool_list = k
    elif isinstance(k[0], (int, np.integer)):  # list of integers
        bool_list = [False] * n
        for i, k_i in enumerate(k):
            if k_i is not False:
                # if k_i is True:
                #    bool_list[i] = True
                if k_i < n:
                    bool_list[k_i] = True
                else:
                    raise ValueError("k as set by %s is not convertable in a bool string!" % k)
    else:
        raise ValueError('input list k as %s not compatible' % k)
    return bool_list
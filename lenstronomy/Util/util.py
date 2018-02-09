__author__ = 'Simon Birrer'

"""
this file contains standard routines
"""

import numpy as np
import mpmath


def map_coord2pix(ra, dec, x_0, y_0, M):
    """
    this routines performs a linear transformation between two coordinate systems. Mainly used to transform angular
    into pixel coordinates in an image
    :param ra: ra coordinates
    :param dec: dec coordinates
    :param x_0: pixel value in x-axis of ra,dec = 0,0
    :param y_0: pixel value in y-axis of ra,dec = 0,0
    :param M: 2x2 matrix to transform angular to pixel coordinates
    :return: transformed coordnate systems of input ra and dec
    """
    x, y = M.dot(np.array([ra, dec]))
    return x + x_0, y + y_0


def array2image(array, nx=0, ny=0):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    if nx == 0 or ny == 0:
        n = int(np.sqrt(len(array)))
        if n**2 != len(array):
            raise ValueError("lenght of input array given as %s is not square of integer number!" %(len(array)))
        nx, ny = n, n
    image = array.reshape(int(nx), int(ny))
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh


def make_grid(numPix, deltapix, subgrid_res=1, left_lower=False):
    """

    :param numPix: number of pixels per axis
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if left_lower is True:
        x_grid = matrix[:, 0]*deltapix
        y_grid = matrix[:, 1]*deltapix
    else:
        x_grid = (matrix[:, 0] - (numPix_eff-1)/2.)*deltapix_eff
        y_grid = (matrix[:, 1] - (numPix_eff-1)/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    return x_grid - shift, y_grid - shift


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


def make_grid_with_coordtransform(numPix, deltapix, subgrid_res=1, left_lower=False):
    """
    same as make_grid routine, but returns the transformaton matrix and shift between coordinates and pixel

    :param numPix:
    :param deltapix:
    :param subgrid_res:
    :param left_lower:
    :return:
    """
    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    if left_lower is True:
        x_grid = matrix[:, 0]*deltapix
        y_grid = matrix[:, 1]*deltapix
    else:
        x_grid = (matrix[:, 0] - (numPix_eff-1)/2.)*deltapix_eff
        y_grid = (matrix[:, 1] - (numPix_eff-1)/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    x_grid -= shift
    y_grid -= shift
    ra_at_xy_0 = x_grid[0]
    dec_at_xy_0 = y_grid[0]
    x_at_radec_0 = (numPix_eff-1)/2.
    y_at_radec_0 = (numPix_eff - 1) / 2.
    Mpix2coord = np.array([[deltapix_eff, 0], [0, deltapix_eff]])
    Mcoord2pix = np.linalg.inv(Mpix2coord)
    return x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix


def grid_from_coordinate_transform(numPix, Mpix2coord, ra_at_xy_0, dec_at_xy_0):
    """
    return a grid in x and y coordinates that satisfy the coordinate system


    :param numPix:
    :param Mpix2coord:
    :param ra_at_xy_0:
    :param dec_at_xy_0:
    :return:
    """
    a = np.arange(numPix)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = matrix[:, 0]
    y_grid = matrix[:, 1]
    ra_grid = x_grid * Mpix2coord[0, 0] + y_grid * Mpix2coord[0, 1] + ra_at_xy_0
    dec_grid = x_grid * Mpix2coord[1, 0] + y_grid * Mpix2coord[1, 1] + dec_at_xy_0
    return ra_grid, dec_grid


def get_axes(x, y):
    """
    computes the axis x and y of a given 2d grid
    :param x:
    :param y:
    :return:
    """
    n=int(np.sqrt(len(x)))
    if n**2 != len(x):
        raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(x)))
    x_image = x.reshape(n,n)
    y_image = y.reshape(n,n)
    x_axes = x_image[0,:]
    y_axes = y_image[:,0]
    return x_axes, y_axes


def averaging(grid, numGrid, numPix):
    """
    resize 2d pixel grid with numGrid to numPix and averages over the pixels
    :param grid: higher resolution pixel grid
    :param numGrid:
    :param numPix:
    :return:
    """

    Nbig = int(numGrid)
    Nsmall = int(numPix)
    small = grid.reshape([Nsmall, int(Nbig/Nsmall), Nsmall, int(Nbig/Nsmall)]).mean(3).mean(1)
    return small


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
    absmapped = np.sqrt(x_mapped**2+y_mapped**2)
    return absmapped


def get_distance(x_mins, y_mins, x_true, y_true):
    """

    :param x_mins:
    :param y_mins:
    :param x_true:
    :param y_true:
    :return:
    """
    if len(x_mins) != len(x_true):
        return 10**10
    dist = 0
    x_true_list = np.array(x_true)
    y_true_list = np.array(y_true)

    for i in range(0,len(x_mins)):
        dist_list = (x_mins[i] - x_true_list)**2 + (y_mins[i] - y_true_list)**2
        dist += min(dist_list)
        k = np.where(dist_list == min(dist_list))
        if type(k) != int:
            k = k[0]
        x_true_list = np.delete(x_true_list, k)
        y_true_list = np.delete(y_true_list, k)
    return dist


def compare_distance(x_mapped, y_mapped):
    """

    :param x_mapped: array of x-positions of remapped catalogue image
    :param y_mapped: array of y-positions of remapped catalogue image
    :return: sum of distance square of positions
    """
    X2 = 0
    for i in range(0, len(x_mapped)-1):
        for j in range(i+1, len(x_mapped)):
            dx = x_mapped[i]-x_mapped[j]
            dy = y_mapped[i]-y_mapped[j]
            X2 += dx**2+dy**2
    return X2


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
        dist[i] = np.min((x_1[i] - x_2)**2 + (y_1[i] - y_2)**2)
    return dist


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
        result = array_sorted[n-numSelect:]
    else:
        result = array_sorted[0:numSelect]
    return result[::-1]


def points_on_circle(radius, points):
    """
    returns a set of uniform points around a circle
    :param radius: radius of the circle
    :param points: number of points on the circle
    :return:
    """
    angle = np.linspace(0, 2*np.pi, points)
    x_coord = np.cos(angle)*radius
    y_coord = np.sin(angle)*radius
    return x_coord, y_coord


def neighborSelect(a, x, y):
    """
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
    for i in range(dim+1,len(a)-dim-1):
        if (a[i] < a[i-1]
            and a[i] < a[i+1]
            and a[i] < a[i-dim]
            and a[i] < a[i+dim]
            and a[i] < a[i-(dim-1)]
            and a[i] < a[i-(dim+1)]
            and a[i] < a[i+(dim-1)]
            and a[i] < a[i+(dim+1)]):
                if(a[i] < a[(i-2*dim-1)%dim**2]
                    and a[i] < a[(i-2*dim+1)%dim**2]
                    and a[i] < a[(i-dim-2)%dim**2]
                    and a[i] < a[(i-dim+2)%dim**2]
                    and a[i] < a[(i+dim-2)%dim**2]
                    and a[i] < a[(i+dim+2)%dim**2]
                    and a[i] < a[(i+2*dim-1)%dim**2]
                    and a[i] < a[(i+2*dim+1)%dim**2]):
                    if(a[i] < a[(i-3*dim-1)%dim**2]
                        and a[i] < a[(i-3*dim+1)%dim**2]
                        and a[i] < a[(i-dim-3)%dim**2]
                        and a[i] < a[(i-dim+3)%dim**2]
                        and a[i] < a[(i+dim-3)%dim**2]
                        and a[i] < a[(i+dim+3)%dim**2]
                        and a[i] < a[(i+3*dim-1)%dim**2]
                        and a[i] < a[(i+3*dim+1)%dim**2]):
                        x_mins.append(x[i])
                        y_mins.append(y[i])
                        values.append(a[i])
    return np.array(x_mins), np.array(y_mins), np.array(values)


def fwhm2sigma(fwhm):
    """

    :param fwhm: full-widt-half-max value
    :return: gaussian sigma (sqrt(var))
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return sigma


def sigma2fwhm(sigma):
    """

    :param sigma:
    :return:
    """
    fwhm = sigma * (2 * np.sqrt(2 * np.log(2)))
    return fwhm


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

    ra_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
    dec_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
    for i in range(0, subgrid_res):
        for j in range(0, subgrid_res):
            ra_array_new[i::subgrid_res, j::subgrid_res] = ra_array + d_ra_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_ra_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))
            dec_array_new[i::subgrid_res, j::subgrid_res] = dec_array + d_dec_x * (-1/2. + 1/(2.*subgrid_res) + j/float(subgrid_res)) + d_dec_y * (-1/2. + 1/(2.*subgrid_res) + i/float(subgrid_res))

    ra_coords_sub = image2array(ra_array_new)
    dec_coords_sub = image2array(dec_array_new)
    return ra_coords_sub, dec_coords_sub


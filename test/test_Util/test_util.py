__author__ = 'sibirrer'

import lenstronomy.Util.util as Util

import numpy as np
import pytest
import numpy.testing as npt


def test_map_coord2pix():
    ra = 0
    dec = 0
    x_0 = 1
    y_0 = -1
    M = np.array([[1, 0], [0, 1]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x == 1
    assert y == -1

    ra = [0, 1, 2]
    dec = [0, 2, 1]
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[0] == 1
    assert y[0] == -1
    assert x[1] == 2

    M = np.array([[0, 1], [1, 0]])
    x, y = Util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[1] == 3
    assert y[1] == 0


def test_make_grid():
    numPix = 11
    deltapix = 1.
    grid = Util.make_grid(numPix, deltapix)
    assert grid[0][0] == -5
    assert np.sum(grid[0]) == 0
    x_grid, y_grid = Util.make_grid(numPix, deltapix, subgrid_res=2.)
    print(np.sum(x_grid))
    assert np.sum(x_grid) == 0
    assert x_grid[0] == -5.25


def test_make_grid_transform():
    numPix = 11
    theta = np.pi / 2
    deltaPix = 0.05
    Mpix2coord = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * deltaPix
    ra_coord, dec_coord = Util.make_grid_transformed(numPix, Mpix2coord)
    ra2d = Util.array2image(ra_coord)
    assert ra2d[5, 5] == 0
    assert ra2d[4, 5] == deltaPix
    npt.assert_almost_equal(ra2d[5, 4], 0, decimal=10)


def test_grid_with_coords():
    numPix = 11
    deltaPix = 1.
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    ra = 0
    dec = 0
    x, y = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = .1
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    ra = 0
    dec = 0
    x, y = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = 1.
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    x_, y_ = 0, 0
    ra, dec = Util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == -5
    assert dec == -5

    numPix = 11
    deltaPix = .1
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = Util.make_grid_with_coordtransform(numPix, deltaPix, subgrid_res=1, left_lower=False)
    x_, y_ = 0, 0
    ra, dec = Util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == -.5
    assert dec == -.5
    x__, y__ = Util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x__ == x_
    assert y__ == y_


def test_array2image():
    array = np.linspace(1, 100, 100)
    image = Util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10


def test_image2array():
    image = np.zeros((10,10))
    image[1,2] = 1
    array = Util.image2array(image)
    assert array[12] == 1


def test_image2array2image():
    image = np.zeros((20, 10))
    nx, ny = np.shape(image)
    image[1, 2] = 1
    array = Util.image2array(image)
    image_new = Util.array2image(array, nx, ny)
    assert image_new[1, 2] == image[1, 2]


def test_get_axes():
    numPix = 11
    deltapix = 0.1
    x_grid, y_grid = Util.make_grid(numPix,deltapix)
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    npt.assert_almost_equal(x_axes[0], -0.5, decimal=12)
    npt.assert_almost_equal(y_axes[0], -0.5, decimal=12)
    npt.assert_almost_equal(x_axes[1], -0.4, decimal=12)
    npt.assert_almost_equal(y_axes[1], -0.4, decimal=12)
    x_grid += 1
    x_axes, y_axes = Util.get_axes(x_grid, y_grid)
    npt.assert_almost_equal(x_axes[0], 0.5, decimal=12)
    npt.assert_almost_equal(y_axes[0], -0.5, decimal=12)


def test_symmetry():
    array = np.linspace(0,10,100)
    image = Util.array2image(array)
    array_new = Util.image2array(image)
    assert array_new[42] == array[42]


def test_displaceAbs():
    x = np.array([0,1,2])
    y = np.array([3,2,1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = Util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    assert result[0] == np.sqrt(2)
    assert result[1] == 0


def test_get_distance():
    x_mins = np.array([1.])
    y_mins = np.array([1.])
    x_true = np.array([0.])
    y_true = np.array([0.])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2

    x_mins = np.array([1.,2])
    y_mins = np.array([1.,1])
    x_true = np.array([0.])
    y_true = np.array([0.])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 10000000000

    x_mins = np.array([1.,2])
    y_mins = np.array([1.,1])
    x_true = np.array([0.,1])
    y_true = np.array([0.,2])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 6

    x_mins = np.array([1.,2,0])
    y_mins = np.array([1.,1,0])
    x_true = np.array([0.,1,1])
    y_true = np.array([0.,2,1])
    dist = Util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2


def test_selectBest():
    array = np.array([4,3,6,1,3])
    select = np.array([2,4,7,3,3])
    numSelect = 4
    array_select = Util.selectBest(array, select, numSelect, highest=True)
    assert array_select[0] == 6
    assert array_select[3] == 1


def test_compare_distance():
    x_mapped = np.array([4,3,6,1,3])
    y_mapped = np.array([2,4,7,3,3])
    X2 = Util.compare_distance(x_mapped, y_mapped)
    assert X2 == 140


def test_min_square_dist():
    x_1 = np.array([4, 3, 6, 1, 3])
    y_1 = np.array([2, 4, 7, 3, 3])
    x_2 = np.array([4, 3, 6, 1, 3])
    y_2 = np.array([2, 3, 7, 3, 3])
    dist = Util.min_square_dist(x_1, y_1, x_2, y_2)
    assert dist[0] == 0
    assert dist[1] == 1


def test_neighborSelect():
    a = np.ones(100)
    a[41] = 0
    x = np.linspace(0,99,100)
    y = np.linspace(0,99,100)
    x_mins, y_mins, values = Util.neighborSelect(a, x, y)
    assert x_mins[0] == 41
    assert y_mins[0] == 41
    assert values[0] == 0


def test_make_subgrid():
    numPix = 101
    deltapix = 1
    x_grid, y_grid = Util.make_grid(numPix, deltapix, subgrid_res=1)
    x_sub_grid, y_sub_grid = Util.make_subgrid(x_grid, y_grid, subgrid_res=2)
    assert np.sum(x_grid) == 0
    assert x_sub_grid[0] == -50.25
    assert y_sub_grid[17] == -50.25

    x_sub_grid_new, y_sub_grid_new = Util.make_subgrid(x_grid, y_grid, subgrid_res=4)
    assert x_sub_grid_new[0] == -50.375


def test_fwhm2sigma():
    fwhm = 0.5
    sigma = Util.fwhm2sigma(fwhm)
    assert sigma == fwhm/ (2 * np.sqrt(2 * np.log(2)))


def test_points_on_circle():
    radius = 1
    points = 8
    ra, dec = Util.points_on_circle(radius, points)
    assert ra[0] == 1
    assert dec[0] == 0


if __name__ == '__main__':
    pytest.main()

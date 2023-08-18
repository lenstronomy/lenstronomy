__author__ = "sibirrer"

import lenstronomy.Util.util as util

import numpy as np
import pytest
import numpy.testing as npt
import unittest


def test_estimate_theta_E():
    x = np.array([-0.45328229, 0.57461556, 0.53757501, -0.42312438])
    y = np.array([0.69582971, -0.51226356, 0.37577509, -0.40245467])

    approx = util.approx_theta_E(x, y)
    npt.assert_array_less(approx - 1, 0.2)


def test_sort_img_index():
    ximg, yimg = np.array([1, 2, 3, 4]), np.array([0, 0, 1, 2])

    xref, yref = np.array([2, 3, 1, 4]), np.array([0, 1, 0, 2])

    indexes = util.sort_image_index(ximg, yimg, xref, yref)

    npt.assert_allclose(ximg[indexes], xref)
    npt.assert_allclose(yimg[indexes], yref)

    xref, yref = np.array([2, 3, 1, 4]), np.array([0, 1, 0, 2])

    indexes = util.sort_image_index(xref, yref, xref, yref)

    npt.assert_allclose(np.array(indexes), [0, 1, 2, 3])


def test_map_coord2pix():
    ra = 0
    dec = 0
    x_0 = 1
    y_0 = -1
    M = np.array([[1, 0], [0, 1]])
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x == 1
    assert y == -1

    ra = [0, 1, 2]
    dec = [0, 2, 1]
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[0] == 1
    assert y[0] == -1
    assert x[1] == 2

    M = np.array([[0, 1], [1, 0]])
    x, y = util.map_coord2pix(ra, dec, x_0, y_0, M)
    assert x[1] == 3
    assert y[1] == 0


def test_make_grid():
    numPix = 11
    deltapix = 1.0

    grid = util.make_grid(numPix, deltapix)
    assert grid[0][0] == -5
    assert np.sum(grid[0]) == 0.0

    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=2.0)
    assert np.sum(x_grid) == 0.0
    assert x_grid[0] == -5.25

    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=1, left_lower=True)
    assert x_grid[0] == 0.0
    assert y_grid[0] == 0.0

    # Similar tests for a non-rectangular grid

    x_grid, y_grid = util.make_grid((numPix, numPix - 1), deltapix)
    assert x_grid[0] == -5.0
    assert y_grid[0] == -4.5
    assert np.sum(x_grid) == np.sum(y_grid) == 0

    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=2.0)
    assert np.sum(x_grid) == np.sum(y_grid) == 0
    assert x_grid[0] == -5.25

    x_grid, y_grid = util.make_grid(numPix, deltapix, left_lower=True)
    assert x_grid[0] == 0
    assert y_grid[0] == 0


def test_make_grid_transform():
    numPix = 11
    theta = np.pi / 2
    deltaPix = 0.05
    Mpix2coord = (
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        * deltaPix
    )
    ra_coord, dec_coord = util.make_grid_transformed(numPix, Mpix2coord)
    ra2d = util.array2image(ra_coord)
    assert ra2d[5, 5] == 0
    assert ra2d[4, 5] == deltaPix
    npt.assert_almost_equal(ra2d[5, 4], 0, decimal=10)


def test_grid_with_coords():
    numPix = 11
    deltaPix = 1.0
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=False
    )
    ra = 0
    dec = 0
    x, y = util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = 0.1
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=False
    )
    ra = 0
    dec = 0
    x, y = util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x == 5
    assert y == 5

    numPix = 11
    deltaPix = 1.0
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=False, inverse=True
    )
    x_, y_ = 0, 0
    ra, dec = util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == 5
    assert dec == -5

    numPix = 11
    deltaPix = 1.0
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=False, inverse=False
    )
    x_, y_ = 0, 0
    ra, dec = util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == -5
    assert dec == -5

    numPix = 11
    deltaPix = 0.1
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=False
    )
    x_, y_ = 0, 0
    ra, dec = util.map_coord2pix(x_, y_, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    assert ra == 0.5
    assert dec == -0.5
    x__, y__ = util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x__ == x_
    assert y__ == y_

    numPix = 11
    deltaPix = 0.1
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=True
    )
    assert ra_at_xy_0 == 0
    assert dec_at_xy_0 == 0

    numPix = 11
    deltaPix = 0.1
    (
        x_grid,
        y_grid,
        ra_at_xy_0,
        dec_at_xy_0,
        x_at_radec_0,
        y_at_radec_0,
        Mpix2coord,
        Mcoord2pix,
    ) = util.make_grid_with_coordtransform(
        numPix, deltaPix, subgrid_res=1, left_lower=True, center_ra=2, center_dec=3
    )
    assert ra_at_xy_0 == 2
    assert dec_at_xy_0 == 3


def test_array2image():
    array = np.linspace(1, 100, 100)
    image = util.array2image(array)
    assert image[9][9] == 100
    assert image[0][9] == 10


def test_image2array():
    image = np.zeros((10, 10))
    image[1, 2] = 1
    array = util.image2array(image)
    assert array[12] == 1


def test_image2array2image():
    image = np.zeros((20, 10))
    nx, ny = np.shape(image)
    image[1, 2] = 1
    array = util.image2array(image)
    image_new = util.array2image(array, nx, ny)
    assert image_new[1, 2] == image[1, 2]


def test_array2cube():
    array = np.linspace(1, 200, 200)
    image = util.array2cube(array, 2, 100)
    assert image[0][9][9] == 100
    assert image[1][0][9] == 110


def test_cube2array():
    sube = np.zeros((2, 10, 10))
    sube[1, 2, 2] = 1
    array = util.cube2array(sube)
    assert array[122] == 1


def test_cube2array2cube():
    cube = np.zeros((2, 10, 10))
    ns, nx, ny = np.shape(cube)
    assert nx == ny  # condition required
    nxy = nx * ny
    cube[1, 2, 2] = 1
    array = util.cube2array(cube)
    cube_new = util.array2cube(array, ns, nxy)
    assert cube_new[1, 2, 2] == cube[1, 2, 2]


def test_get_axes():
    numPix = 11
    deltapix = 0.1
    x_grid, y_grid = util.make_grid(numPix, deltapix)
    x_axes, y_axes = util.get_axes(x_grid, y_grid)
    npt.assert_almost_equal(x_axes[0], -0.5, decimal=12)
    npt.assert_almost_equal(y_axes[0], -0.5, decimal=12)
    npt.assert_almost_equal(x_axes[1], -0.4, decimal=12)
    npt.assert_almost_equal(y_axes[1], -0.4, decimal=12)
    x_grid += 1
    x_axes, y_axes = util.get_axes(x_grid, y_grid)
    npt.assert_almost_equal(x_axes[0], 0.5, decimal=12)
    npt.assert_almost_equal(y_axes[0], -0.5, decimal=12)


def test_symmetry():
    array = np.linspace(0, 10, 100)
    image = util.array2image(array)
    array_new = util.image2array(image)
    assert array_new[42] == array[42]


def test_displaceAbs():
    x = np.array([0, 1, 2])
    y = np.array([3, 2, 1])
    sourcePos_x = 1
    sourcePos_y = 2
    result = util.displaceAbs(x, y, sourcePos_x, sourcePos_y)
    assert result[0] == np.sqrt(2)
    assert result[1] == 0


def test_get_distance():
    x_mins = np.array([1.0])
    y_mins = np.array([1.0])
    x_true = np.array([0.0])
    y_true = np.array([0.0])
    dist = util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2

    x_mins = np.array([1.0, 2])
    y_mins = np.array([1.0, 1])
    x_true = np.array([0.0])
    y_true = np.array([0.0])
    dist = util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 10000000000

    x_mins = np.array([1.0, 2])
    y_mins = np.array([1.0, 1])
    x_true = np.array([0.0, 1])
    y_true = np.array([0.0, 2])
    dist = util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 6

    x_mins = np.array([1.0, 2, 0])
    y_mins = np.array([1.0, 1, 0])
    x_true = np.array([0.0, 1, 1])
    y_true = np.array([0.0, 2, 1])
    dist = util.get_distance(x_mins, y_mins, x_true, y_true)
    assert dist == 2


def test_selectBest():
    array = np.array([4, 3, 6, 1, 3])
    select = np.array([2, 4, 7, 3, 3])
    numSelect = 4
    array_select = util.selectBest(array, select, numSelect, highest=True)
    assert array_select[0] == 6
    assert array_select[3] == 1

    array_select = util.selectBest(array, select, numSelect, highest=False)
    assert array_select[0] == 3
    assert array_select[3] == 4

    array_select = util.selectBest(array, select, numSelect=10, highest=False)
    assert len(array_select) == len(array)


def test_select_best():
    array = np.array([4, 3, 6, 1, 3])
    select = np.array([2, 4, 7, 3, 3])
    numSelect = 4
    array_select = util.select_best(array, select, numSelect, highest=True)

    assert array_select[0] == 6
    assert array_select[3] == 1

    array_select = util.select_best(array, select, numSelect, highest=False)
    assert array_select[0] == 3
    assert array_select[3] == 4

    array_select = util.select_best(array, select, num_select=10, highest=False)
    assert len(array_select) == len(array)


def test_compare_distance():
    x_mapped = np.array([4, 3, 6, 1, 3])
    y_mapped = np.array([2, 4, 7, 3, 3])
    X2 = util.compare_distance(x_mapped, y_mapped)
    assert X2 == 140


def test_min_square_dist():
    x_1 = np.array([4, 3, 6, 1, 3])
    y_1 = np.array([2, 4, 7, 3, 3])
    x_2 = np.array([4, 3, 6, 1, 3])
    y_2 = np.array([2, 3, 7, 3, 3])
    dist = util.min_square_dist(x_1, y_1, x_2, y_2)
    assert dist[0] == 0
    assert dist[1] == 1


def test_neighbor_select_fast():
    a = np.ones(100)
    a[41] = 0
    x = np.linspace(0, 99, 100)
    y = np.linspace(0, 99, 100)
    x_mins, y_mins, values = util.local_minima_2d(a, x, y)
    assert x_mins[0] == 41
    assert y_mins[0] == 41
    assert values[0] == 0


def test_neighborSelect():
    a = np.ones(100)
    a[41] = 0
    x = np.linspace(0, 99, 100)
    y = np.linspace(0, 99, 100)
    x_mins, y_mins, values = util.neighborSelect(a, x, y)
    assert x_mins[0] == 41
    assert y_mins[0] == 41
    assert values[0] == 0


def test_make_subgrid():
    numPix = 101
    deltapix = 1
    x_grid, y_grid = util.make_grid(numPix, deltapix, subgrid_res=1)
    x_sub_grid, y_sub_grid = util.make_subgrid(x_grid, y_grid, subgrid_res=2)
    assert np.sum(x_grid) == 0
    assert len(x_grid) == 101 * 101
    assert x_sub_grid[0] == -50.25
    assert y_sub_grid[17] == -50.25

    x_sub_grid_new, y_sub_grid_new = util.make_subgrid(x_grid, y_grid, subgrid_res=4)
    assert x_sub_grid_new[0] == -50.375


def test_fwhm2sigma():
    fwhm = 0.5
    sigma = util.fwhm2sigma(fwhm)
    assert sigma == fwhm / (2 * np.sqrt(2 * np.log(2)))


def test_points_on_circle():
    radius = 1
    points = 8
    ra, dec = util.points_on_circle(radius, points, connect_ends=True)
    assert ra[0] == 1
    assert dec[0] == 0

    ra_, dec_ = util.points_on_circle(radius, points - 1, connect_ends=False)
    npt.assert_almost_equal(ra[:-1], ra_, decimal=8)
    npt.assert_almost_equal(dec[:-1], dec_, decimal=8)


def test_convert_bool_list():
    bool_list = util.convert_bool_list(n=10, k=None)
    assert len(bool_list) == 10
    assert bool_list[0] == True

    bool_list = util.convert_bool_list(n=10, k=3)
    assert len(bool_list) == 10
    assert bool_list[3] is True
    assert bool_list[2] is False

    bool_list = util.convert_bool_list(n=10, k=[3, 7])
    assert len(bool_list) == 10
    assert bool_list[3] is True
    assert bool_list[7] is True
    assert bool_list[2] is False

    bool_list = util.convert_bool_list(n=3, k=[False, False, True])
    assert len(bool_list) == 3
    assert bool_list[0] is False
    assert bool_list[1] is False
    assert bool_list[2] is True

    bool_list = util.convert_bool_list(n=3, k=[])
    assert len(bool_list) == 3
    assert bool_list[0] is False


def test_area():
    r = 1
    x_, y_ = util.points_on_circle(radius=r, connect_ends=True, num_points=1000)
    vs = np.dstack([x_, y_])[0]
    a = util.area(vs)
    npt.assert_almost_equal(a, np.pi * r**2, decimal=3)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            array = np.ones(5)
            util.array2image(array)
        with self.assertRaises(ValueError):
            array = np.ones((2, 2))
            util.array2cube(array, 2, 2)
        with self.assertRaises(ValueError):
            x, y = np.ones(6), np.ones(6)
            util.get_axes(x, y)
        with self.assertRaises(ValueError):
            util.selectBest(
                array=np.ones(6), criteria=np.ones(5), numSelect=1, highest=True
            )
        with self.assertRaises(ValueError):
            util.select_best(
                array=np.ones(6), criteria=np.ones(5), num_select=1, highest=True
            )
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=2, k=[3, 7])
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=3, k=[True, True])
        with self.assertRaises(ValueError):
            util.convert_bool_list(n=2, k=[0.1, True])

    def test_raise_make_grid(self):
        with self.assertRaises(ValueError):
            util.make_grid(numPix=1.1, deltapix=1)
        with self.assertRaises(ValueError):
            util.make_grid(numPix=[1.1, 1], deltapix=1)


if __name__ == "__main__":
    pytest.main()

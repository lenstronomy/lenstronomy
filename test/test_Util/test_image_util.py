__author__ = 'sibirrer'

import lenstronomy.Util.util as util
import pytest
import unittest
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.image_util as image_util


def test_add_layer2image_odd_odd():
    grid2d = np.zeros((101, 101))
    kernel = np.zeros((21, 21))
    kernel[10, 10] = 1
    x_pos = 50
    y_pos = 50
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[50, 50] == 1
    assert added[49, 49] == 0

    x_pos = 70
    y_pos = 95
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)

    assert added[95, 70] == 1

    x_pos = 20
    y_pos = 45
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[45, 20] == 1

    x_pos = 45
    y_pos = 20
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[20, 45] == 1

    x_pos = 20
    y_pos = 55
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[55, 20] == 1

    x_pos = 20
    y_pos = 100
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=0)
    assert added[100, 20] == 1

    x_pos = 20.5
    y_pos = 100
    added = image_util.add_layer2image(grid2d, x_pos, y_pos, kernel, order=1)
    assert added[100, 20] == 0.5
    assert added[100, 21] == 0.5


def test_add_layer2image_int():
    grid2d = np.zeros((7, 7))
    x_pos, y_pos = 4, 1
    kernel = np.ones((3, 3))
    added = image_util.add_layer2image_int(grid2d, x_pos, y_pos, kernel)
    print(added)
    assert added[0, 0] == 0
    assert added[0, 3] == 1

    added = image_util.add_layer2image_int(grid2d, x_pos + 10, y_pos, kernel)
    print(added)
    npt.assert_almost_equal(grid2d, added, decimal=9)


def test_add_background():
    image = np.ones((10, 10))
    sigma_bkgd = 1.
    image_noisy = image_util.add_background(image, sigma_bkgd)
    assert abs(np.sum(image_noisy)) < np.sqrt(np.sum(image)*sigma_bkgd)*3


def test_add_poisson():
    image = np.ones((100, 100))
    exp_time = 100.
    poisson = image_util.add_poisson(image, exp_time)
    assert abs(np.sum(poisson)) < np.sqrt(np.sum(image)/exp_time)*10


def test_findOverlap():
    x_mins = [0,1,0]
    y_mins = [1,2,1]
    deltapix = 0.5
    x_mins, y_mins = image_util.findOverlap(x_mins, y_mins, deltapix)
    print(x_mins, y_mins)
    assert x_mins[0] == 0
    assert y_mins[0] == 1
    assert len(x_mins) == 2


def test_coordInImage():
    x_coord = [100,20,-10]
    y_coord = [0,-30,5]
    numPix = 50
    deltapix = 1
    x_result, y_result = image_util.coordInImage(x_coord, y_coord, numPix, deltapix)
    assert x_result == -10
    assert y_result == 5


def test_rebin_coord_transform():
    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=3, deltapix=0.03, subgrid_res=1)
    x_grid, y_grid, ra_at_xy_0_re, dec_at_xy_0_re, x_at_radec_0_re, y_at_radec_0_re, Mpix2coord_re, Mcoord2pix_re = util.make_grid_with_coordtransform(numPix=1, deltapix=0.09, subgrid_res=1)

    ra_at_xy_0_resized, dec_at_xy_0_resized, x_at_radec_0_resized, y_at_radec_0_resized, Mpix2coord_resized, Mcoord2pix_resized = image_util.rebin_coord_transform(3, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix)
    assert ra_at_xy_0_resized == ra_at_xy_0_re
    assert dec_at_xy_0_resized == dec_at_xy_0_re
    assert x_at_radec_0_resized == x_at_radec_0_re
    assert y_at_radec_0_resized == y_at_radec_0_re
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)

    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=100, deltapix=0.05, subgrid_res=1)
    x_grid, y_grid, ra_at_xy_0_re, dec_at_xy_0_re, x_at_radec_0_re, y_at_radec_0_re, Mpix2coord_re, Mcoord2pix_re = util.make_grid_with_coordtransform(numPix=50, deltapix=0.1, subgrid_res=1)

    ra_at_xy_0_resized, dec_at_xy_0_resized, x_at_radec_0_resized, y_at_radec_0_resized, Mpix2coord_resized, Mcoord2pix_resized = image_util.rebin_coord_transform(2, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix)
    assert ra_at_xy_0_resized == ra_at_xy_0_re
    assert dec_at_xy_0_resized == dec_at_xy_0_re
    assert x_at_radec_0_resized == x_at_radec_0_re
    assert y_at_radec_0_resized == y_at_radec_0_re
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)

    x_grid, y_grid, ra_at_xy_0, dec_at_xy_0, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix = util.make_grid_with_coordtransform(numPix=99, deltapix=0.1, subgrid_res=1)
    x_grid, y_grid, ra_at_xy_0_re, dec_at_xy_0_re, x_at_radec_0_re, y_at_radec_0_re, Mpix2coord_re, Mcoord2pix_re = util.make_grid_with_coordtransform(numPix=33, deltapix=0.3, subgrid_res=1)

    assert x_at_radec_0 == 49
    ra_at_xy_0_resized, dec_at_xy_0_resized, x_at_radec_0_resized, y_at_radec_0_resized, Mpix2coord_resized, Mcoord2pix_resized = image_util.rebin_coord_transform(3, x_at_radec_0, y_at_radec_0, Mpix2coord, Mcoord2pix)

    assert x_at_radec_0_resized == 16
    npt.assert_almost_equal(ra_at_xy_0_resized, ra_at_xy_0_re, decimal=8)
    npt.assert_almost_equal(dec_at_xy_0_resized, dec_at_xy_0_re, decimal=8)
    npt.assert_almost_equal(x_at_radec_0_resized, x_at_radec_0_re, decimal=8)
    npt.assert_almost_equal(y_at_radec_0_resized, y_at_radec_0_re, decimal=8)
    npt.assert_almost_equal(Mcoord2pix_resized[0][0], Mcoord2pix_re[0][0], decimal=8)
    npt.assert_almost_equal(Mpix2coord_re[0][0], Mpix2coord_resized[0][0], decimal=8)

    x_in, y_in = 10., 10.
    ra, dec = util.map_coord2pix(x_in, y_in, ra_at_xy_0, dec_at_xy_0, Mpix2coord)
    x_out, y_out = util.map_coord2pix(ra, dec, x_at_radec_0, y_at_radec_0, Mcoord2pix)
    assert x_in == x_out
    assert y_in == y_out

    x_in, y_in = 10., 10.
    ra, dec = util.map_coord2pix(x_in, y_in, ra_at_xy_0_resized, dec_at_xy_0_resized, Mpix2coord_resized)
    x_out, y_out = util.map_coord2pix(ra, dec, x_at_radec_0_resized, y_at_radec_0_resized, Mcoord2pix_resized)
    assert x_in == x_out
    assert y_in == y_out


def test_rotateImage():
    img = np.zeros((5, 5))
    img[2, 2] = 1
    img[1, 2] = 0.5

    angle = 360
    im_rot = image_util.rotateImage(img, angle)
    npt.assert_almost_equal(im_rot[1, 2], 0.5, decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0., decimal=10)

    angle = 360./2
    im_rot = image_util.rotateImage(img, angle)
    npt.assert_almost_equal(im_rot[1, 2], 0., decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[3, 2], 0.5, decimal=10)

    angle = 360./4
    im_rot = image_util.rotateImage(img, angle)
    npt.assert_almost_equal(im_rot[1, 2], 0., decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0.5, decimal=10)

    angle = 360./8
    im_rot = image_util.rotateImage(img, angle)
    npt.assert_almost_equal(im_rot[1, 2], 0.23931518624017051, decimal=10)
    npt.assert_almost_equal(im_rot[2, 2], 1., decimal=10)
    npt.assert_almost_equal(im_rot[2, 1], 0.23931518624017073, decimal=10)


def test_re_size_array():
    numPix = 9
    kernel = np.zeros((numPix, numPix))
    kernel[int((numPix-1)/2), int((numPix-1)/2)] = 1
    subgrid_res = 2
    input_values = kernel
    x_in = np.linspace(0, 1, numPix)
    x_out = np.linspace(0, 1, numPix*subgrid_res)
    out_values = image_util.re_size_array(x_in, x_in, input_values, x_out, x_out)
    kernel_out = out_values
    assert kernel_out[int((numPix*subgrid_res-1)/2), int((numPix*subgrid_res-1)/2)] == 0.58477508650519028


def test_symmetry_average():
    image = np.zeros((5,5))
    image[2, 3] = 1
    symmetry = 2
    img_sym = image_util.symmetry_average(image, symmetry)
    npt.assert_almost_equal(img_sym[2, 1], 0.5, decimal=10)


def test_cut_edges():
    image = np.zeros((51,51))
    image[25][25] = 1
    numPix = 21
    resized = image_util.cut_edges(image, numPix)
    nx, ny = resized.shape
    assert nx == numPix
    assert ny == numPix
    assert resized[10][10] == 1


def test_re_size():
    grid = np.zeros((200, 100))
    grid[100, 50] = 4
    grid_small = image_util.re_size(grid, factor=2)
    assert grid_small[50][25] == 1


def test_stack_images():
    numPix = 10
    image1 = np.ones((numPix, numPix))
    image2 = np.ones((numPix, numPix)) / 10.
    image_list = [image1, image2]
    wht1 = np.ones((numPix, numPix))
    wht2 = np.ones((numPix, numPix)) * 10
    wht_list = [wht1, wht2]
    sigma_list = [0.1, 0.2]
    image_stacked, wht_stacked, sigma_stacked = image_util.stack_images(image_list=image_list, wht_list=wht_list, sigma_list=sigma_list)
    assert sigma_stacked == 0.19306145983268458
    assert image_stacked[0, 0] == 0.18181818181818182
    assert wht_stacked[0, 0] == 5.5


def test_rebin_image():
    numPix = 10
    bin_size = 2
    image = np.ones((numPix, numPix))
    wht_map = np.ones((numPix, numPix)) * 10
    idex_mask = np.ones((numPix, numPix))
    sigma_bkg = 0.1
    ra_coords, dec_coords = util.make_grid(numPix, deltapix=0.05)
    ra_coords = util.array2image(ra_coords)
    dec_coords = util.array2image(dec_coords)
    image_resized, wht_map_resized, sigma_bkg_resized, ra_coords_resized, dec_coords_resized, idex_mask_resized = image_util.rebin_image(bin_size, image, wht_map, sigma_bkg, ra_coords, dec_coords, idex_mask)
    assert image_resized[0, 0] == 4
    assert wht_map_resized[0, 0] == wht_map[0, 0]
    assert sigma_bkg_resized == 0.2
    assert ra_coords_resized[0, 0] == -0.2

    numPix = 11
    bin_size = 2
    image = np.ones((numPix, numPix))
    wht_map = np.ones((numPix, numPix)) * 10
    idex_mask = np.ones((numPix, numPix))
    sigma_bkg = 0.1
    ra_coords, dec_coords = util.make_grid(numPix, deltapix=0.05)
    ra_coords = util.array2image(ra_coords)
    dec_coords = util.array2image(dec_coords)
    image_resized, wht_map_resized, sigma_bkg_resized, ra_coords_resized, dec_coords_resized, idex_mask_resized = image_util.rebin_image(
        bin_size, image, wht_map, sigma_bkg, ra_coords, dec_coords, idex_mask)
    assert image_resized[0, 0] == 4
    assert wht_map_resized[0, 0] == wht_map[0, 0]
    assert sigma_bkg_resized == 0.2
    npt.assert_almost_equal(ra_coords_resized[0, 0], -0.225, decimal=8)


def test_radial_profile():
    from lenstronomy.LightModel.Profiles.gaussian import Gaussian
    gauss = Gaussian()
    x, y = util.make_grid(11, 1)
    flux = gauss.function(x, y, sigma_x=10, sigma_y=10, amp=1)
    data = util.array2image(flux)
    profile_r = image_util.radial_profile(data, center=[5, 5])
    profile_r_true = gauss.function(np.linspace(0, stop=7, num=8), 0, sigma_x=10, sigma_y=10, amp=1)
    npt.assert_almost_equal(profile_r, profile_r_true, decimal=3)


def cut_edges():
    image = np.zeros((5, 5))
    image[2, 2] = 1
    numPix = 3
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image_cut[1, 1] == 1

    image = np.zeros((6, 6))
    image[3, 2] = 1
    numPix = 4
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image[2, 1] == 1

    image = np.zeros((6, 8))
    image[3, 2] = 1
    numPix = 4
    image_cut = image_util.cut_edges(image, numPix)
    assert len(image_cut) == numPix
    assert image[2, 0] == 1


class TestRaise(unittest.TestCase):

    def test_raise(self):

        with self.assertRaises(ValueError):
            grid2d = np.zeros((7, 7))
            x_pos, y_pos = 4, 1
            kernel = np.ones((2, 2))
            added = image_util.add_layer2image_int(grid2d, x_pos, y_pos, kernel)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.re_size(image, factor=2)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.re_size(image, factor=0.5)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.cut_edges(image, numPix=7)
        with self.assertRaises(ValueError):
            image = np.ones((5, 6))
            image_util.cut_edges(image, numPix=3)
        with self.assertRaises(ValueError):
            image = np.ones((5, 5))
            image_util.cut_edges(image, numPix=2)


if __name__ == '__main__':
    pytest.main()

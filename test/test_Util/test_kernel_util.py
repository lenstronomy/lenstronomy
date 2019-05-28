import lenstronomy.Util.util as Util
import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.gaussian import Gaussian
gaussian = Gaussian()
import pytest
import unittest
import numpy as np
import numpy.testing as npt
import scipy.ndimage.interpolation as interp


def test_fwhm_kernel():
    x_grid, y_gird = Util.make_grid(101, 1)
    sigma = 20

    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel = Util.array2image(flux)
    kernel = kernel_util.kernel_norm(kernel)
    fwhm_kernel = kernel_util.fwhm_kernel(kernel)
    fwhm = Util.sigma2fwhm(sigma)
    npt.assert_almost_equal(fwhm/fwhm_kernel, 1, 2)


def test_center_kernel():
    x_grid, y_gird = Util.make_grid(31, 1)
    sigma = 2
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel = Util.array2image(flux)
    kernel = kernel_util.kernel_norm(kernel)

    # kernel being centered
    kernel_new = kernel_util.center_kernel(kernel, iterations=20)
    kernel_new = kernel_util.kernel_norm(kernel_new)
    npt.assert_almost_equal(kernel_new/kernel, 1, decimal=8)
    # kernel shifted in x
    kernel_shifted = interp.shift(kernel, [-.1, 0], order=1)
    kernel_new = kernel_util.center_kernel(kernel_shifted, iterations=5)
    kernel_new = kernel_util.kernel_norm(kernel_new)
    npt.assert_almost_equal((kernel_new + 0.00001) / (kernel + 0.00001), 1, decimal=4)
    # kernel shifted in y
    kernel_shifted = interp.shift(kernel, [0, -0.4], order=1)
    kernel_new = kernel_util.center_kernel(kernel_shifted, iterations=5)
    kernel_new = kernel_util.kernel_norm(kernel_new)
    npt.assert_almost_equal((kernel_new + 0.01) / (kernel + 0.01), 1, decimal=3)
    # kernel shifted in x and y
    kernel_shifted = interp.shift(kernel, [0.2, -0.3], order=1)
    kernel_new = kernel_util.center_kernel(kernel_shifted, iterations=5)
    kernel_new = kernel_util.kernel_norm(kernel_new)
    npt.assert_almost_equal((kernel_new + 0.01) / (kernel + 0.01), 1, decimal=3)


def test_pixelsize_change():
    kernel = np.zeros((7, 7))
    kernel[3, 3] = 1
    deltaPix_in = 0.1
    deltaPix_out = 0.2
    kernel_new = kernel_util.kernel_pixelsize_change(kernel, deltaPix_in, deltaPix_out)
    assert len(kernel_new) == 3
    assert kernel_new[1, 1] == 1


def test_cutout_source():
    """
    test whether a shifted psf can be reproduced sufficiently well
    :return:
    """
    kernel_size = 5
    image = np.zeros((10, 10))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 1
    shift_x = 0.5
    shift_y = 0
    x_c, y_c = 5, 5
    x_pos = x_c + shift_x
    y_pos = y_c + shift_y
    #kernel_shifted = interp.shift(kernel, [shift_y, shift_x], order=1)
    image = image_util.add_layer2image(image, x_pos, y_pos, kernel, order=1)
    print(image)
    kernel_new = kernel_util.cutout_source(x_pos=x_pos, y_pos=y_pos, image=image, kernelsize=kernel_size)
    npt.assert_almost_equal(kernel_new[2, 2], kernel[2, 2], decimal=2)


def test_cutout_source_border():
    kernel_size = 7
    image = np.zeros((10, 10))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 1
    shift_x = +0.1
    shift_y = 0
    x_c, y_c = 2, 5
    x_pos = x_c + shift_x
    y_pos = y_c + shift_y
    #kernel_shifted = interp.shift(kernel, [shift_y, shift_x], order=1)
    image = image_util.add_layer2image(image, x_pos, y_pos, kernel, order=1)
    kernel_new = kernel_util.cutout_source(x_pos=x_pos, y_pos=y_pos, image=image, kernelsize=kernel_size)
    nx_new, ny_new = np.shape(kernel_new)
    print(kernel_new)
    assert nx_new == kernel_size
    assert ny_new == kernel_size
    npt.assert_almost_equal(kernel_new[2, 2], kernel[2, 2], decimal=2)


def test_cut_psf():
    image = np.ones((7, 7))
    psf_cut = kernel_util.cut_psf(image, 5)
    assert len(psf_cut) == 5


def test_de_shift():
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 2
    shift_x = 0.48
    shift_y = 0.2
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    kernel_de_shifted = kernel_util.de_shift_kernel(kernel_shifted, shift_x, shift_y, iterations=50)
    delta_max = np.max(kernel- kernel_de_shifted)
    assert delta_max < 0.01
    npt.assert_almost_equal(kernel_de_shifted[2, 2], kernel[2, 2], decimal=2)

    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[2, 2] = 2
    shift_x = 1.48
    shift_y = 0.2
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    kernel_de_shifted = kernel_util.de_shift_kernel(kernel_shifted, shift_x, shift_y, iterations=50)
    delta_max = np.max(kernel - kernel_de_shifted)
    assert delta_max < 0.01
    npt.assert_almost_equal(kernel_de_shifted[2, 2], kernel[2, 2], decimal=2)

    kernel_size_x = 5
    kernel_size_y = 4
    kernel = np.zeros((kernel_size_x, kernel_size_y))
    kernel[2, 2] = 2
    shift_x = 1.48
    shift_y = 0.2
    kernel_shifted = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    kernel_de_shifted = kernel_util.de_shift_kernel(kernel_shifted, shift_x, shift_y, iterations=50)
    delta_max = np.max(kernel - kernel_de_shifted)
    assert delta_max < 0.01
    npt.assert_almost_equal(kernel_de_shifted[2, 2], kernel[2, 2], decimal=2)


def test_deshift_subgrid():
    # test the de-shifting with a sharpened subgrid kernel
    kernel_size = 5
    subgrid = 3
    fwhm = 1
    kernel_subgrid_size = kernel_size * subgrid
    kernel_subgrid = np.zeros((kernel_subgrid_size, kernel_subgrid_size))
    kernel_subgrid[7, 7] = 2
    kernel_subgrid = kernel_util.kernel_gaussian(kernel_subgrid_size, 1./subgrid, fwhm=fwhm)

    kernel = util.averaging(kernel_subgrid, kernel_subgrid_size, kernel_size)

    shift_x = 0.18
    shift_y = 0.2
    shift_x_subgird = shift_x * subgrid
    shift_y_subgrid = shift_y * subgrid
    kernel_shifted_subgrid = interp.shift(kernel_subgrid, [-shift_y_subgrid, -shift_x_subgird], order=1)
    kernel_shifted = util.averaging(kernel_shifted_subgrid, kernel_subgrid_size, kernel_size)
    kernel_shifted_highres = kernel_util.subgrid_kernel(kernel_shifted, subgrid_res=subgrid, num_iter=1)
    #npt.assert_almost_equal(kernel_shifted_highres[7, 7], kernel_shifted_subgrid[7, 7], decimal=10)


def test_shift_long_dist():
    """
    input is a shifted kernel by more than 1 pixel
    :return:
    """

    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[4, 4] = 2.
    shift_x = 2.
    shift_y = 1.
    input_kernel = interp.shift(kernel, [-shift_y, -shift_x], order=1)
    old_style_kernel = interp.shift(input_kernel, [shift_y, shift_x], order=1)
    shifted_new = kernel_util.de_shift_kernel(input_kernel, shift_x, shift_y)
    assert kernel[3, 2] == shifted_new[3, 2]
    assert np.max(old_style_kernel - shifted_new) < 0.01


def test_pixel_kernel():
    # point source kernel
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[4, 4] = 1.
    pixel_kernel = kernel_util.pixel_kernel(point_source_kernel=kernel, subgrid_res=1)
    assert pixel_kernel[4, 4] == kernel[4, 4]

    pixel_kernel = kernel_util.pixel_kernel(point_source_kernel=kernel, subgrid_res=11)
    npt.assert_almost_equal(pixel_kernel[4, 4], 0.65841, decimal=3)


def test_split_kernel():
    kernel = np.zeros((9, 9))
    kernel[4, 4] = 1
    subgrid_res = 3
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    subsampling_size = 3
    kernel_hole, kernel_cutout = kernel_util.split_kernel(subgrid_kernel, supersampling_kernel_size=subsampling_size,
                                                          supersampling_factor=subgrid_res)

    assert kernel_hole[4, 4] == 0
    assert len(kernel_cutout) == subgrid_res*subsampling_size
    npt.assert_almost_equal(np.sum(kernel_hole) + np.sum(kernel_cutout), 1, decimal=4)

    subgrid_res = 2
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    subsampling_size = 3
    kernel_hole, kernel_cutout = kernel_util.split_kernel(subgrid_kernel, supersampling_kernel_size=subsampling_size,
                                                          supersampling_factor=subgrid_res)

    assert kernel_hole[4, 4] == 0
    assert len(kernel_cutout) == subgrid_res * subsampling_size + 1
    npt.assert_almost_equal(np.sum(kernel_hole) + np.sum(kernel_cutout), 1, decimal=4)


def test_cutout_source2():
    grid2d = np.zeros((20, 20))
    grid2d[7:9, 7:9] = 1
    kernel = kernel_util.cutout_source(x_pos=7.5, y_pos=7.5, image=grid2d, kernelsize=5, shift=False)
    assert kernel[2, 2] == 1


def test_subgrid_kernel():

    kernel = np.zeros((9, 9))
    kernel[4, 4] = 1
    subgrid_res = 3
    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    kernel_re_sized = image_util.re_size(subgrid_kernel, factor=subgrid_res) *subgrid_res**2
    #import matplotlib.pyplot as plt
    #plt.matshow(kernel); plt.show()
    #plt.matshow(subgrid_kernel); plt.show()
    #plt.matshow(kernel_re_sized);plt.show()
    #plt.matshow(kernel_re_sized- kernel);plt.show()
    npt.assert_almost_equal(kernel_re_sized[4, 4], 1, decimal=2)
    assert np.max(subgrid_kernel) == subgrid_kernel[13, 13]
    #assert kernel_re_sized[4, 4] == 1


def test_subgrid_rebin():
    kernel_size = 11
    subgrid_res = 3

    sigma = 1
    x_grid, y_gird = Util.make_grid(kernel_size, 1./subgrid_res, subgrid_res)
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel = Util.array2image(flux)
    print(np.shape(kernel))
    kernel = util.averaging(kernel, numGrid=kernel_size * subgrid_res, numPix=kernel_size)
    kernel = kernel_util.kernel_norm(kernel)

    subgrid_kernel = kernel_util.subgrid_kernel(kernel, subgrid_res=subgrid_res, odd=True)
    kernel_pixel = util.averaging(subgrid_kernel, numGrid=kernel_size * subgrid_res, numPix=kernel_size)
    kernel_pixel = kernel_util.kernel_norm(kernel_pixel)
    assert np.sum((kernel_pixel - kernel)**2) < 0.1


def test_mge_kernel():
    from lenstronomy.LightModel.Profiles.gaussian import MultiGaussian
    mg = MultiGaussian()
    fraction_list = [0.2, 0.7, 0.1]
    sigmas_scaled = [5, 10, 15]
    x, y = util.make_grid(numPix=101, deltapix=1)
    kernel = mg.function(x, y, amp=fraction_list, sigma=sigmas_scaled)
    kernel = util.array2image(kernel)

    amps, sigmas, norm = kernel_util.mge_kernel(kernel, order=10)
    print(amps, sigmas, norm)
    kernel_new = mg.function(x, y, amp=amps, sigma=sigmas)
    kernel_new = util.array2image(kernel_new)
    #npt.assert_almost_equal(sigmas_scaled, sigmas)
    #npt.assert_almost_equal(amps, fraction_list)
    npt.assert_almost_equal(kernel_new, kernel, decimal=3)


def test_kernel_average_pixel():
    gaussian = Gaussian()
    subgrid_res = 3
    x_grid, y_gird = Util.make_grid(9, 1., subgrid_res)
    sigma = 2
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel_super = Util.array2image(flux)
    kernel_pixel = kernel_util.kernel_average_pixel(kernel_super, supersampling_factor=subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_pixel), np.sum(kernel_super))

    kernel_pixel = kernel_util.kernel_average_pixel(kernel_super, supersampling_factor=2)
    npt.assert_almost_equal(np.sum(kernel_pixel), np.sum(kernel_super))


def test_averaging_even_kernel():

    subgrid_res = 4

    x_grid, y_gird = Util.make_grid(19, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel_super = Util.array2image(flux)

    kernel_pixel = kernel_util.averaging_even_kernel(kernel_super, subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_pixel), 1, decimal=5)
    assert len(kernel_pixel) == 5

    x_grid, y_gird = Util.make_grid(17, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel_super = Util.array2image(flux)

    kernel_pixel = kernel_util.averaging_even_kernel(kernel_super, subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_pixel), 1, decimal=5)
    assert len(kernel_pixel) == 5


def test_degrade_kernel():
    subgrid_res = 2

    x_grid, y_gird = Util.make_grid(19, 1., 1)
    sigma = 1.5
    flux = gaussian.function(x_grid, y_gird, amp=1, sigma_x=sigma, sigma_y=sigma)
    kernel_super = Util.array2image(flux)/np.sum(flux)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=subgrid_res)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=3)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)

    kernel_degraded = kernel_util.degrade_kernel(kernel_super, degrading_factor=1)
    npt.assert_almost_equal(np.sum(kernel_degraded), 1, decimal=8)


class TestRaise(unittest.TestCase):

    def test_raise(self):
        with self.assertRaises(ValueError):
            kernel = np.zeros((2, 2))
            kernel_util.center_kernel(kernel, iterations=1)

        with self.assertRaises(ValueError):
            kernel_super = np.ones((9, 9))
            kernel_util.split_kernel(kernel_super, supersampling_kernel_size=2, supersampling_factor=3)
        with self.assertRaises(ValueError):
            kernel_util.split_kernel(kernel_super, supersampling_kernel_size=3, supersampling_factor=0)
        with self.assertRaises(ValueError):
            image = np.ones((10, 10))
            kernel_util.cutout_source(x_pos=3, y_pos=2, image=image, kernelsize=2)
        with self.assertRaises(ValueError):
            kernel_util.fwhm_kernel(kernel=np.ones((4, 4)))
        with self.assertRaises(ValueError):
            kernel_util.fwhm_kernel(kernel=np.ones((5, 5)))


if __name__ == '__main__':
    pytest.main()

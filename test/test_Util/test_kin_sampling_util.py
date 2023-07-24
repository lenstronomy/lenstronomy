import lenstronomy.Util.kin_sampling_util as kin_sampling_util

import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt


class TestKinSamplingUtil(object):
    def setup(self):
        self.image = np.array([[0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

        transform_pix2angle = np.array([[1, 0], [0, 1]])  # linear translation matrix of no rotation
        ra_at_xy_0, dec_at_xy_0 = transform_pix2angle.dot([-5 / 2, -5 / 2]).T
        self.kwargs_pixel_rot_hst = {'nx': 5, 'ny': 5,  # number of pixels per axis
                                     'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                                     'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                                     'transform_pix2angle': transform_pix2angle}
        self.hst_inputs = {'image': self.image, 'transform_pix2angle': self.kwargs_pixel_rot_hst['transform_pix2angle'],
                           'ra_at_xy0': self.kwargs_pixel_rot_hst['ra_at_xy_0'],
                           'dec_at_xy0': self.kwargs_pixel_rot_hst['dec_at_xy_0'],
                           'ellipse_PA': 0,
                           'offset_x': 0, 'offset_y': 0}
        self.kinNN_inputs = {'image': self.image, 'deltaPix': 1}

    def define_muse_kwargs_from_rotation_angle(self, rotation_angle):
        transform_pix2angle = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle),
                                                                                            np.cos(
                                                                                                rotation_angle)]])  # linear translation matrix of a 90 degree rotation
        ra_at_xy_0, dec_at_xy_0 = transform_pix2angle.dot([-5 / 2, -5 / 2]).T
        kwargs_pixel_rot_muse = {'nx': 5, 'ny': 5,  # number of pixels per axis
                                 'ra_at_xy_0': ra_at_xy_0,  # RA at pixel (0,0)
                                 'dec_at_xy_0': dec_at_xy_0,  # DEC at pixel (0,0)
                                 'transform_pix2angle': transform_pix2angle}
        muse_inputs = {'image': self.image, 'transform_pix2angle': kwargs_pixel_rot_muse['transform_pix2angle'],
                       'ra_at_xy0': kwargs_pixel_rot_muse['ra_at_xy_0'],
                       'dec_at_xy0': kwargs_pixel_rot_muse['dec_at_xy_0']}
        return muse_inputs

    def test_rotation(self):
        # muse grid not rotated from HST:
        muse_inputs = self.define_muse_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(muse_inputs, self.hst_inputs, self.kinNN_inputs)
        muse_img_coords = skinn_align.musegrid_in_hstxy()
        hst_img_coords = skinn_align.pix_coords(self.hst_inputs)
        npt.assert_allclose(hst_img_coords, muse_img_coords, atol=1e-7)

        muse_inputs = self.define_muse_kwargs_from_rotation_angle(2 * np.pi)
        skinn_align = kin_sampling_util.KinNNImageAlign(muse_inputs, self.hst_inputs, self.kinNN_inputs)
        muse_img_coords = skinn_align.musegrid_in_hstxy()
        hst_img_coords = skinn_align.pix_coords(self.hst_inputs)
        npt.assert_allclose(hst_img_coords, muse_img_coords, atol=1e-7)

        # muse_grid 90 degrees from HST:
        muse_inputs = self.define_muse_kwargs_from_rotation_angle(np.pi / 2)
        skinn_align = kin_sampling_util.KinNNImageAlign(muse_inputs, self.hst_inputs, self.kinNN_inputs)
        muse_img_coords = skinn_align.musegrid_in_hstxy()
        hst_img_coords = skinn_align.pix_coords(self.hst_inputs)
        npt.assert_allclose(hst_img_coords[0], muse_img_coords[1], atol=1e-7)  # old x coordinate is new y coordinate

    def test_samegrid(self):
        # check interpolation in case where grids are the same
        muse_inputs = self.define_muse_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(muse_inputs, self.hst_inputs, self.kinNN_inputs)
        interp_image = skinn_align.interp_image()
        npt.assert_almost_equal(interp_image, self.image)

    def test_offset(self):
        hst_inputs = {'image': self.image, 'transform_pix2angle': self.kwargs_pixel_rot_hst['transform_pix2angle'],
                      'ra_at_xy0': self.kwargs_pixel_rot_hst['ra_at_xy_0'],
                      'dec_at_xy0': self.kwargs_pixel_rot_hst['dec_at_xy_0'],
                      'ellipse_PA': 0,
                      'offset_x': 1, 'offset_y': 1}
        muse_inputs = self.define_muse_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(muse_inputs, hst_inputs, self.kinNN_inputs)
        interp_image = skinn_align.interp_image()
        # #uncomment to see plots
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(interp_image)
        # plt.show()
        expected_img = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        npt.assert_allclose(interp_image, expected_img, atol=1e-7)

import lenstronomy.Util.kin_sampling_util as kin_sampling_util

import numpy as np
import pytest
import numpy.testing as npt
import matplotlib.pyplot as plt


class TestKinSamplingUtil(object):
    def setup(self):
        self.image = np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        transform_pix2angle = np.array(
            [[1, 0], [0, 1]]
        )  # linear translation matrix of no rotation
        ra_at_xy_0, dec_at_xy_0 = transform_pix2angle.dot([-5 / 2, -5 / 2]).T
        self.kwargs_pixel_rot_imaging = {
            "nx": 5,
            "ny": 5,  # number of pixels per axis
            "ra_at_xy_0": ra_at_xy_0,  # RA at pixel (0,0)
            "dec_at_xy_0": dec_at_xy_0,  # DEC at pixel (0,0)
            "transform_pix2angle": transform_pix2angle,
        }
        self.imaging_inputs = {
            "image": self.image,
            "transform_pix2angle": self.kwargs_pixel_rot_imaging["transform_pix2angle"],
            "ra_at_xy0": self.kwargs_pixel_rot_imaging["ra_at_xy_0"],
            "dec_at_xy0": self.kwargs_pixel_rot_imaging["dec_at_xy_0"],
            "ellipse_PA": 0,
            "offset_x": 0,
            "offset_y": 0,
        }
        self.kinNN_inputs = {"image": self.image, "deltaPix": 1}

    def define_spectra_kwargs_from_rotation_angle(self, rotation_angle):
        transform_pix2angle = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )  # linear translation matrix of a 90 degree rotation
        ra_at_xy_0, dec_at_xy_0 = transform_pix2angle.dot([-5 / 2, -5 / 2]).T
        kwargs_pixel_rot_spectra = {
            "nx": 5,
            "ny": 5,  # number of pixels per axis
            "ra_at_xy_0": ra_at_xy_0,  # RA at pixel (0,0)
            "dec_at_xy_0": dec_at_xy_0,  # DEC at pixel (0,0)
            "transform_pix2angle": transform_pix2angle,
        }
        spectra_inputs = {
            "image": self.image,
            "transform_pix2angle": kwargs_pixel_rot_spectra["transform_pix2angle"],
            "ra_at_xy0": kwargs_pixel_rot_spectra["ra_at_xy_0"],
            "dec_at_xy0": kwargs_pixel_rot_spectra["dec_at_xy_0"],
        }
        return spectra_inputs

    def test_rotation(self):
        # spectra grid not rotated from imaging:
        spectra_inputs = self.define_spectra_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(
            spectra_inputs, self.imaging_inputs, self.kinNN_inputs
        )
        spectra_img_coords = skinn_align.spectragrid_in_imagingxy()
        imaging_img_coords = skinn_align.pix_coords(self.imaging_inputs)
        npt.assert_allclose(imaging_img_coords, spectra_img_coords, atol=1e-7)

        spectra_inputs = self.define_spectra_kwargs_from_rotation_angle(2 * np.pi)
        skinn_align = kin_sampling_util.KinNNImageAlign(
            spectra_inputs, self.imaging_inputs, self.kinNN_inputs
        )
        spectra_img_coords = skinn_align.spectragrid_in_imagingxy()
        imaging_img_coords = skinn_align.pix_coords(self.imaging_inputs)
        npt.assert_allclose(imaging_img_coords, spectra_img_coords, atol=1e-7)

        # spectra_grid 90 degrees from imaging:
        spectra_inputs = self.define_spectra_kwargs_from_rotation_angle(np.pi / 2)
        skinn_align = kin_sampling_util.KinNNImageAlign(
            spectra_inputs, self.imaging_inputs, self.kinNN_inputs
        )
        spectra_img_coords = skinn_align.spectragrid_in_imagingxy()
        imaging_img_coords = skinn_align.pix_coords(self.imaging_inputs)
        npt.assert_allclose(
            imaging_img_coords[0], spectra_img_coords[1], atol=1e-7
        )  # old x coordinate is new y coordinate

    def test_samegrid(self):
        # check interpolation in case where grids are the same
        spectra_inputs = self.define_spectra_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(
            spectra_inputs, self.imaging_inputs, self.kinNN_inputs
        )
        interp_image = skinn_align.interp_image()
        npt.assert_almost_equal(interp_image, self.image)

    def test_offset(self):
        imaging_inputs = {
            "image": self.image,
            "transform_pix2angle": self.kwargs_pixel_rot_imaging["transform_pix2angle"],
            "ra_at_xy0": self.kwargs_pixel_rot_imaging["ra_at_xy_0"],
            "dec_at_xy0": self.kwargs_pixel_rot_imaging["dec_at_xy_0"],
            "ellipse_PA": 0,
            "offset_x": 1,
            "offset_y": 1,
        }
        spectra_inputs = self.define_spectra_kwargs_from_rotation_angle(0)
        skinn_align = kin_sampling_util.KinNNImageAlign(
            spectra_inputs, imaging_inputs, self.kinNN_inputs
        )
        interp_image = skinn_align.interp_image()
        # #uncomment to see plots
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(interp_image)
        # plt.show()
        expected_img = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        npt.assert_allclose(interp_image, expected_img, atol=1e-7)

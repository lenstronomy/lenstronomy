__author__ = "dangilman"

import numpy.testing as npt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_lens_model,
    setup_grids,
    coordinates_and_deflections,
    class_setup,
)
import numpy as np
import pytest


class TestMultiPlaneDecoupled(object):
    def setup_method(self):
        self.zlens = 0.5
        self.z_source = 2.0
        self.kwargs_lens_true = [
            {
                "theta_E": 1.0,
                "center_x": 0.0,
                "center_y": 0.0,
                "e1": 0.2,
                "e2": -0.1,
                "gamma": 2.0,
            },
            {"theta_E": 0.2, "center_x": 0.6, "center_y": 0.3},
            {"theta_E": 0.2, "center_x": -0.6, "center_y": -1.0},
        ]
        self.lens_model_list = ["EPL", "SIS", "SIS"]
        self.lens_redshift_list = [self.zlens, 0.25, 1.0]
        self.lens_model_true = LensModel(
            self.lens_model_list,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source=self.z_source,
        )
        self.cosmo = self.lens_model_true.cosmo

    def test_zsplit_requirements(self):
        """
        TODO: generalize to an arbitrary set of deflectors at different redshifts with deflection angles decoupled from
        the line of sight
        """
        index_lens_split = [0, 1]
        args = (self.lens_model_true, self.kwargs_lens_true, index_lens_split)
        npt.assert_raises(Exception, setup_lens_model, args)

    def test_setup_lens_model(self):
        index_lens_split = [0]
        (
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            z_source,
            z_split,
            cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, index_lens_split
        )
        npt.assert_equal(len(lens_model_fixed.lens_model_list), 2)
        npt.assert_equal(len(lens_model_free.lens_model_list), 1)
        npt.assert_equal(lens_model_fixed.redshift_list[0], 0.25)
        npt.assert_equal(lens_model_fixed.redshift_list[1], 1.0)
        npt.assert_equal(lens_model_free.redshift_list[0], 0.5)
        npt.assert_equal(z_source, 2.0)
        npt.assert_equal(z_split, 0.5)
        npt.assert_equal(kwargs_lens_free[0]["theta_E"], 1.0)
        npt.assert_equal(kwargs_lens_fixed[0]["theta_E"], 0.2)

    def test_setup_grids(self):
        grid_size = 2.0
        grid_resolution = 0.001
        xx, yy, interp_points, npixels = setup_grids(grid_size, grid_resolution)
        npt.assert_equal(npixels, grid_size / grid_resolution)
        npt.assert_equal(xx[0], -1)
        npt.assert_equal(yy[0], -1)
        npt.assert_equal(interp_points[0][0], -1)
        npt.assert_equal(interp_points[1][0], -1)
        npt.assert_equal(interp_points[0][-1], 1)
        npt.assert_equal(interp_points[1][-1], 1)

    def test_coordinates_and_deflections(self):
        index_lens_split = [0]
        (
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            z_source,
            z_split,
            cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, index_lens_split
        )

        x_coordinate_arcsec = 0.9
        y_coordinate_arcsec = -0.5
        (
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
        ) = coordinates_and_deflections(
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            x_coordinate_arcsec,
            y_coordinate_arcsec,
            z_split,
            z_source,
            cosmo_bkg,
        )
        (
            x_true,
            y_true,
            alpha_x_foreground_true,
            alpha_y_foreground_true,
        ) = self.lens_model_true.lens_model.ray_shooting_partial(
            0.0,
            0.0,
            x_coordinate_arcsec,
            y_coordinate_arcsec,
            0.0,
            z_split,
            self.kwargs_lens_true,
        )
        npt.assert_almost_equal(x, x_true)
        npt.assert_almost_equal(y, y_true)

        Td = cosmo_bkg.T_xy(0, z_split)
        theta_x, theta_y = x_true / Td, y_true / Td
        (
            alpha_x_main,
            alpha_y_main,
        ) = self.lens_model_true.lens_model._multi_plane_base.func_list[0].derivatives(
            theta_x, theta_y, **self.kwargs_lens_true[0]
        )
        reduced_to_physical = cosmo_bkg.d_xy(0.0, self.z_source) / cosmo_bkg.d_xy(
            z_split, z_source
        )
        alpha_x_main_physical = alpha_x_main * reduced_to_physical
        alpha_y_main_physical = alpha_y_main * reduced_to_physical
        npt.assert_almost_equal(
            alpha_x_foreground, alpha_x_foreground_true + alpha_x_main_physical
        )
        npt.assert_almost_equal(
            alpha_y_foreground, alpha_y_foreground_true + alpha_y_main_physical
        )

        source_x_true, source_y_true = self.lens_model_true.ray_shooting(
            x_coordinate_arcsec, y_coordinate_arcsec, self.kwargs_lens_true
        )

        alpha_x = alpha_x_foreground - alpha_x_main_physical
        alpha_y = alpha_y_foreground - alpha_y_main_physical
        Tds = cosmo_bkg.T_xy(z_split, z_source)
        Ts = cosmo_bkg.T_xy(0, z_source)
        # this defines alpha_beta_subx and alpha_beta_suby; don't totally understand the sign here
        source_x = (x + (alpha_x + alpha_beta_subx) * Tds) / Ts
        source_y = (y + (alpha_y + alpha_beta_suby) * Tds) / Ts
        npt.assert_almost_equal(source_x, source_x_true, 5)
        npt.assert_almost_equal(source_y, source_y_true, 5)

    def test_class_setup(self):
        index_lens_split = [0]
        (
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            z_source,
            z_split,
            cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, index_lens_split
        )

        # TEST POINT
        coordinate_type = "POINT"
        x_coordinate_arcsec_point = 0.9
        y_coordinate_arcsec_point = -0.5
        source_x_true, source_y_true = self.lens_model_true.ray_shooting(
            x_coordinate_arcsec_point, y_coordinate_arcsec_point, self.kwargs_lens_true
        )
        (
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
        ) = coordinates_and_deflections(
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            x_coordinate_arcsec_point,
            y_coordinate_arcsec_point,
            z_split,
            z_source,
            cosmo_bkg,
        )
        kwargs_class_setup = class_setup(
            lens_model_free,
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
            z_split,
            coordinate_type,
        )
        lens_model_decoupled = LensModel(**kwargs_class_setup)
        beta_x_point, beta_y_point = lens_model_decoupled.ray_shooting(
            x_coordinate_arcsec_point, y_coordinate_arcsec_point, kwargs_lens_free
        )
        npt.assert_almost_equal(beta_x_point, source_x_true)
        npt.assert_almost_equal(beta_y_point, source_y_true)

        # TEST GRID
        coordinate_type = "GRID"
        grid_size = 2.0
        grid_resolution = 0.005
        xgrid, ygrid, interp_points, npixels = setup_grids(grid_size, grid_resolution)
        source_x_true, source_y_true = self.lens_model_true.ray_shooting(
            xgrid, ygrid, self.kwargs_lens_true
        )
        (
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
        ) = coordinates_and_deflections(
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            xgrid,
            ygrid,
            z_split,
            z_source,
            cosmo_bkg,
        )
        kwargs_class_setup = class_setup(
            lens_model_free,
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
            z_split,
            coordinate_type,
            interp_points=interp_points,
        )
        lens_model_decoupled = LensModel(**kwargs_class_setup)
        beta_x, beta_y = lens_model_decoupled.ray_shooting(
            x_coordinate_arcsec_point, y_coordinate_arcsec_point, kwargs_lens_free
        )
        npt.assert_almost_equal(beta_x, beta_x_point, 5)
        npt.assert_almost_equal(beta_y, beta_y_point, 5)

        beta_x_grid, beta_y_grid = lens_model_decoupled.ray_shooting(
            xgrid, ygrid, kwargs_lens_free
        )
        npt.assert_allclose(beta_x_grid, source_x_true, 5)
        npt.assert_allclose(beta_y_grid, source_y_true, 5)

        # TEST MULTIPLE IMAGES
        coordinate_type = "MULTIPLE_IMAGES"
        x_image = np.array([1.0, x_coordinate_arcsec_point, -0.8, 0.0])
        y_image = np.array([0.2, y_coordinate_arcsec_point, 0.9, 2.0])
        source_x_true, source_y_true = self.lens_model_true.ray_shooting(
            x_image, y_image, self.kwargs_lens_true
        )
        (
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
        ) = coordinates_and_deflections(
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            x_image,
            y_image,
            z_split,
            z_source,
            cosmo_bkg,
        )
        kwargs_class_setup = class_setup(
            lens_model_free,
            x,
            y,
            alpha_x_foreground,
            alpha_y_foreground,
            alpha_beta_subx,
            alpha_beta_suby,
            z_split,
            coordinate_type,
            x_image=x_image,
            y_image=y_image,
        )
        lens_model_decoupled = LensModel(**kwargs_class_setup)
        (
            beta_x_multiple_images_array,
            beta_y_multiple_images_array,
        ) = lens_model_decoupled.ray_shooting(x_image, y_image, kwargs_lens_free)
        npt.assert_allclose(beta_x_multiple_images_array, source_x_true, 5)
        npt.assert_allclose(beta_y_multiple_images_array, source_y_true, 5)

        (
            beta_x_multiple_images,
            beta_y_multiple_images,
        ) = lens_model_decoupled.ray_shooting(
            x_image[1] + 0.05, y_image[1] - 0.1, kwargs_lens_free
        )
        npt.assert_almost_equal(beta_x_multiple_images, beta_x_multiple_images_array[1])
        npt.assert_almost_equal(beta_x_multiple_images, beta_x_point)
        npt.assert_almost_equal(beta_y_multiple_images, beta_y_multiple_images_array[1])
        npt.assert_almost_equal(beta_y_multiple_images, beta_y_point)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

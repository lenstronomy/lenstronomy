__author__ = "dangilman"

import numpy.testing as npt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    setup_lens_model,
    setup_grids,
    coordinates_and_deflections,
    setup_raytracing_lensmodels,
    class_setup,
)
from copy import deepcopy
import numpy as np
import pytest


class TestMultiPlaneDecoupled(object):

    def setup_method(self):
        self.zlens = 0.5
        self.z_source = 2.0
        self.kwargs_lens_true = [
            {
                "theta_E": 0.7,
                "center_x": 0.0,
                "center_y": -0.0,
                "e1": 0.2,
                "e2": -0.1,
                "gamma": 2.0,
            },
            {"theta_E": 0.2, "center_x": 0.0, "center_y": -0.4},
            {"theta_E": 0.2, "center_x": 0.6, "center_y": 0.3},
            {"theta_E": 0.15, "center_x": -0.6, "center_y": -1.0},
            {"gamma1": 0.1, "gamma2": -0.2},
        ]
        self.lens_model_list = ["EPL", "SIS", "SIS", "SIS", "SHEAR"]
        self.index_lens_split = [0, 1]
        self.lens_redshift_list = [self.zlens, self.zlens, 0.25, 1.0, 1.5]
        self.lens_model_true = LensModel(
            self.lens_model_list,
            lens_redshift_list=self.lens_redshift_list,
            multi_plane=True,
            z_source=self.z_source,
        )
        self.cosmo = self.lens_model_true.cosmo

        (
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.z_source,
            self.z_split,
            self.cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, self.index_lens_split
        )

        self.Td = self.cosmo_bkg.T_xy(0, self.zlens)
        self.Ts = self.cosmo_bkg.T_xy(0, self.z_source)
        self.Tds = self.cosmo_bkg.T_xy(self.zlens, self.z_source)
        self.reduced_to_phys = self.cosmo_bkg.d_xy(
            0, self.z_source
        ) / self.cosmo_bkg.d_xy(self.zlens, self.z_source)

        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES
        self.x_image = [1.0, 0.5, -0.5, -1.0]
        self.y_image = [-0.4, 0.5, 1.0, 0.7]
        self.x_point = self.x_image[1]
        self.y_point = self.y_image[1]
        self._setup_point()
        self._setup_grid()
        self._setup_multiple_images()

    def _setup_point(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES

        (
            self.x0_point,
            self.y0_point,
            self.alphax_foreground_point,
            self.alphay_foreground_point,
            self.alphax_background_point,
            self.alphay_background_point,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.x_image[1],
            self.y_image[1],
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_point = class_setup(
            self.lens_model_free,
            self.x0_point,
            self.y0_point,
            self.alphax_foreground_point,
            self.alphay_foreground_point,
            self.alphax_background_point,
            self.alphay_background_point,
            self.z_split,
            coordinate_type="POINT",
        )

    def _setup_grid(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES
        grid_size = 2.5
        grid_resolution = 0.005
        self.grid_x, self.grid_y, self.interp_points_grid, self.npix_grid = setup_grids(
            grid_size, grid_resolution, 0.0, 0.0
        )
        (
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.grid_x,
            self.grid_y,
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_grid = class_setup(
            self.lens_model_free,
            self.x0_grid,
            self.y0_grid,
            self.alphax_foreground_grid,
            self.alphay_foreground_grid,
            self.alphax_background_grid,
            self.alphay_background_grid,
            self.z_split,
            coordinate_type="GRID",
            interp_points=self.interp_points_grid,
        )

    def _setup_multiple_images(self):
        # SETUP TESTS FOR POINT, GRID, AND MULTIPLE IMAGES

        (
            self.x0_MI,
            self.y0_MI,
            self.alphax_foreground_MI,
            self.alphay_foreground_MI,
            self.alphax_background_MI,
            self.alphay_background_MI,
        ) = coordinates_and_deflections(
            self.lens_model_fixed,
            self.lens_model_free,
            self.kwargs_lens_fixed,
            self.kwargs_lens_free,
            self.x_image,
            self.y_image,
            self.z_split,
            self.z_source,
            self.cosmo_bkg,
        )

        self.kwargs_multiplane_model_MI = class_setup(
            self.lens_model_free,
            self.x0_MI,
            self.y0_MI,
            self.alphax_foreground_MI,
            self.alphay_foreground_MI,
            self.alphax_background_MI,
            self.alphay_background_MI,
            self.z_split,
            coordinate_type="MULTIPLE_IMAGES",
            x_image=self.x_image,
            y_image=self.y_image,
        )

    def test_zsplit_requirements(self):
        """
        TODO: generalize to an arbitrary set of deflectors at different redshifts with deflection angles decoupled from
        the line of sight
        """
        index_lens_split = [0, 1, 2]
        args = (self.lens_model_true, self.kwargs_lens_true, index_lens_split)
        npt.assert_raises(Exception, setup_lens_model, args)

    def test_setup_grids(self):
        grid_size = 2.0
        grid_resolution = 0.001
        ximg = 0.5
        yimg = -0.2
        xx, yy, interp_points, npixels = setup_grids(
            grid_size,
            grid_resolution,
            coordinate_center_x=ximg,
            coordinate_center_y=yimg,
        )
        npt.assert_equal(npixels - 1, grid_size / grid_resolution)
        npt.assert_equal(xx[0], -1 + ximg)
        npt.assert_equal(yy[0], -1 + yimg)
        npt.assert_equal(interp_points[0][0], -1 + ximg)
        npt.assert_equal(interp_points[1][0], -1 + yimg)
        npt.assert_equal(interp_points[0][-1], 1 + ximg)
        npt.assert_equal(interp_points[1][-1], 1 + yimg)

    def test_setup_lens_model(self):

        (
            lens_model_fixed,
            lens_model_free,
            kwargs_lens_fixed,
            kwargs_lens_free,
            z_source,
            z_split,
            cosmo_bkg,
        ) = setup_lens_model(
            self.lens_model_true, self.kwargs_lens_true, self.index_lens_split
        )
        npt.assert_equal(len(lens_model_fixed.lens_model_list), 3)
        npt.assert_equal(
            len(lens_model_free.lens_model_list), len(self.index_lens_split)
        )
        npt.assert_equal(lens_model_fixed.redshift_list[0], 0.25)
        npt.assert_equal(lens_model_fixed.redshift_list[1], 1.0)
        npt.assert_equal(lens_model_free.redshift_list[0], self.zlens)
        npt.assert_equal(lens_model_free.redshift_list[0], self.zlens)
        npt.assert_equal(z_source, 2.0)
        npt.assert_equal(z_split, self.zlens)
        npt.assert_equal(kwargs_lens_free[0]["theta_E"], 0.7)
        npt.assert_equal(kwargs_lens_free[1]["theta_E"], 0.2)
        npt.assert_equal(kwargs_lens_fixed[0]["theta_E"], 0.2)
        npt.assert_equal(kwargs_lens_fixed[1]["theta_E"], 0.15)

    def test_point_deflection_model(self):

        # the true source coordinate
        x_source_true, y_source_true, _, _ = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                0.0,
                0.0,
                self.x_point,
                self.y_point,
                0.0,
                self.z_source,
                self.kwargs_lens_true,
            )
        )
        beta_x_true, beta_y_true = x_source_true / self.Ts, y_source_true / self.Ts
        npt.assert_almost_equal(
            beta_x_true,
            self.lens_model_true.ray_shooting(
                self.x_point, self.y_point, self.kwargs_lens_true
            )[0],
        )

        # create new dictionary with the macromodel effectively removed
        kwargs_lens_free_no_macromodel = deepcopy(self.kwargs_lens_true)
        kwargs_lens_free_no_macromodel[0]["theta_E"] = 0.0
        kwargs_lens_free_no_macromodel[1]["theta_E"] = 0.0

        # show that the x0, y0, and ray angles are as expected
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                0.0,
                0.0,
                self.x_point,
                self.y_point,
                0.0,
                self.zlens,
                kwargs_lens_free_no_macromodel,
            )
        )
        npt.assert_almost_equal(self.x0_point, x0_true)
        npt.assert_almost_equal(self.y0_point, y0_true)
        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"]["x0_interp"](
                self.x_point, self.y_point
            ),
            x0_true,
        )
        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"]["y0_interp"](
                self.x_point, self.y_point
            ),
            y0_true,
        )
        npt.assert_almost_equal(self.alphax_foreground_point, alphax_fore_true)
        npt.assert_almost_equal(self.alphay_foreground_point, alphay_fore_true)

        # compute the true ray angles at the main lens plane
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                0.0,
                0.0,
                self.x_point,
                self.y_point,
                0.0,
                self.zlens,
                self.kwargs_lens_true,
            )
        )
        npt.assert_almost_equal(self.x0_point, x0_true)
        npt.assert_almost_equal(self.y0_point, y0_true)

        # compute deflections from the macromodel only
        theta_x, theta_y = self.x0_point / self.Td, self.y0_point / self.Td
        alpha_x_macro, alpha_y_macro = 0.0, 0.0
        func_list = self.lens_model_true.lens_model._multi_plane_base.func_list
        for k in range(0, 2):
            _alpha_x_macromodel, _alpha_y_macromodel = func_list[k].derivatives(
                theta_x, theta_y, **self.kwargs_lens_true[k]
            )
            alpha_x_macro += _alpha_x_macromodel * self.reduced_to_phys
            alpha_y_macro += _alpha_y_macromodel * self.reduced_to_phys

        # the full deflection field at the main lens plane is alpha_x_true = alpha_x_foreground - alpha_x_macro
        # therefore alpha_x_foreground = alpha_x_true + alpha_x_macro
        npt.assert_almost_equal(
            self.alphax_foreground_point, alphax_fore_true + alpha_x_macro
        )
        npt.assert_almost_equal(
            self.alphay_foreground_point, alphay_fore_true + alpha_y_macro
        )
        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"][
                "alpha_x_interp_foreground"
            ](self.x_point, self.y_point),
            alphax_fore_true + alpha_x_macro,
        )
        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"][
                "alpha_y_interp_foreground"
            ](self.x_point, self.y_point),
            alphay_fore_true + alpha_y_macro,
        )

        # ray propagation with the small angle approx
        alpha_x = (
            self.alphax_foreground_point - alpha_x_macro + self.alphax_background_point
        )
        alpha_y = (
            self.alphay_foreground_point - alpha_y_macro + self.alphay_background_point
        )
        x_source_point = self.x0_point + alpha_x * self.Tds
        y_source_point = self.y0_point + alpha_y * self.Tds

        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"][
                "alpha_x_interp_background"
            ](self.x_point, self.y_point),
            self.alphax_background_point,
        )
        npt.assert_almost_equal(
            self.kwargs_multiplane_model_point["kwargs_multiplane_model"][
                "alpha_y_interp_background"
            ](self.x_point, self.y_point),
            self.alphay_background_point,
        )

        npt.assert_almost_equal(x_source_true, x_source_point)
        npt.assert_almost_equal(y_source_true, y_source_point)

        lens_model_decoupled = LensModel(**self.kwargs_multiplane_model_point)
        beta_x_new, beta_y_new = lens_model_decoupled.ray_shooting(
            self.x_image[1], self.y_image[1], self.kwargs_lens_free
        )
        beta_x_true, beta_y_true = self.lens_model_true.ray_shooting(
            self.x_image[1], self.y_image[1], self.kwargs_lens_true
        )
        npt.assert_allclose(beta_x_new, beta_x_true)
        npt.assert_allclose(beta_y_new, beta_y_true)

    def test_grid_deflection_model(self):

        # the true source coordinate
        x_source_true, y_source_true, _, _ = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.grid_x),
                np.zeros_like(self.grid_y),
                self.grid_x,
                self.grid_y,
                0.0,
                self.z_source,
                self.kwargs_lens_true,
            )
        )

        # create new dictionary with the macromodel effectively removed
        kwargs_lens_free_no_macromodel = deepcopy(self.kwargs_lens_true)
        kwargs_lens_free_no_macromodel[0]["theta_E"] = 0.0
        kwargs_lens_free_no_macromodel[1]["theta_E"] = 0.0

        # show that the x0, y0, and ray angles are as expected
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.grid_x),
                np.zeros_like(self.grid_y),
                self.grid_x,
                self.grid_y,
                0.0,
                self.zlens,
                kwargs_lens_free_no_macromodel,
            )
        )
        npt.assert_almost_equal(self.x0_grid, x0_true)
        npt.assert_almost_equal(self.y0_grid, y0_true)
        npt.assert_almost_equal(self.alphax_foreground_grid, alphax_fore_true)
        npt.assert_almost_equal(self.alphay_foreground_grid, alphay_fore_true)

        # compute the true ray angles at the main lens plane
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.grid_x),
                np.zeros_like(self.grid_y),
                self.grid_x,
                self.grid_y,
                0.0,
                self.zlens,
                self.kwargs_lens_true,
            )
        )
        npt.assert_almost_equal(self.x0_grid, x0_true)
        npt.assert_almost_equal(self.y0_grid, y0_true)

        # compute deflections from the macromodel only
        theta_x, theta_y = self.x0_grid / self.Td, self.y0_grid / self.Td
        alpha_x_macro, alpha_y_macro = 0.0, 0.0
        func_list = self.lens_model_true.lens_model._multi_plane_base.func_list
        for k in range(0, 2):
            _alpha_x_macromodel, _alpha_y_macromodel = func_list[k].derivatives(
                theta_x, theta_y, **self.kwargs_lens_true[k]
            )
            alpha_x_macro += _alpha_x_macromodel * self.reduced_to_phys
            alpha_y_macro += _alpha_y_macromodel * self.reduced_to_phys

        # the full deflection field at the main lens plane is alpha_x_true = alpha_x_foreground - alpha_x_macro
        # therefore alpha_x_foreground = alpha_x_true + alpha_x_macro
        npt.assert_almost_equal(
            self.alphax_foreground_grid, alphax_fore_true + alpha_x_macro
        )
        npt.assert_almost_equal(
            self.alphay_foreground_grid, alphay_fore_true + alpha_y_macro
        )

        # ray propagation with the small angle approx
        alpha_x = (
            self.alphax_foreground_grid - alpha_x_macro + self.alphax_background_grid
        )
        alpha_y = (
            self.alphay_foreground_grid - alpha_y_macro + self.alphay_background_grid
        )
        x_source_grid = self.x0_grid + alpha_x * self.Tds
        y_source_grid = self.y0_grid + alpha_y * self.Tds

        npt.assert_almost_equal(x_source_true, x_source_grid)
        npt.assert_almost_equal(y_source_true, y_source_grid)

        lens_model_decoupled = LensModel(**self.kwargs_multiplane_model_grid)
        beta_x_new, beta_y_new = lens_model_decoupled.ray_shooting(
            self.x_point, self.y_point, self.kwargs_lens_free
        )
        beta_x_true, beta_y_true = self.lens_model_true.ray_shooting(
            self.x_point, self.y_point, self.kwargs_lens_true
        )
        npt.assert_almost_equal(beta_x_new, beta_x_true)
        npt.assert_almost_equal(beta_y_new, beta_y_true)

    def test_multiple_images_deflection_model(self):

        # the true source coordinate
        x_source_true, y_source_true, _, _ = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.x_image),
                np.zeros_like(self.y_image),
                self.x_image,
                self.y_image,
                0.0,
                self.z_source,
                self.kwargs_lens_true,
            )
        )

        # create new dictionary with the macromodel effectively removed
        kwargs_lens_free_no_macromodel = deepcopy(self.kwargs_lens_true)
        kwargs_lens_free_no_macromodel[0]["theta_E"] = 0.0
        kwargs_lens_free_no_macromodel[1]["theta_E"] = 0.0

        # show that the x0, y0, and ray angles are as expected
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.y_image),
                np.zeros_like(self.x_image),
                self.x_image,
                self.y_image,
                0.0,
                self.zlens,
                kwargs_lens_free_no_macromodel,
            )
        )
        npt.assert_almost_equal(self.x0_MI, x0_true)
        npt.assert_almost_equal(self.y0_MI, y0_true)
        npt.assert_almost_equal(self.alphax_foreground_MI, alphax_fore_true)
        npt.assert_almost_equal(self.alphay_foreground_MI, alphay_fore_true)

        # compute the true ray angles at the main lens plane
        x0_true, y0_true, alphax_fore_true, alphay_fore_true = (
            self.lens_model_true.lens_model.ray_shooting_partial(
                np.zeros_like(self.y_image),
                np.zeros_like(self.x_image),
                self.x_image,
                self.y_image,
                0.0,
                self.zlens,
                self.kwargs_lens_true,
            )
        )
        npt.assert_almost_equal(self.x0_MI, x0_true)
        npt.assert_almost_equal(self.y0_MI, y0_true)

        # compute deflections from the macromodel only
        theta_x, theta_y = self.x0_MI / self.Td, self.y0_MI / self.Td
        alpha_x_macro, alpha_y_macro = 0.0, 0.0
        func_list = self.lens_model_true.lens_model._multi_plane_base.func_list
        for k in range(0, 2):
            _alpha_x_macromodel, _alpha_y_macromodel = func_list[k].derivatives(
                theta_x, theta_y, **self.kwargs_lens_true[k]
            )
            alpha_x_macro += _alpha_x_macromodel * self.reduced_to_phys
            alpha_y_macro += _alpha_y_macromodel * self.reduced_to_phys

        # the full deflection field at the main lens plane is alpha_x_true = alpha_x_foreground - alpha_x_macro
        # therefore alpha_x_foreground = alpha_x_true + alpha_x_macro
        npt.assert_almost_equal(
            self.alphax_foreground_MI, alphax_fore_true + alpha_x_macro
        )
        npt.assert_almost_equal(
            self.alphay_foreground_MI, alphay_fore_true + alpha_y_macro
        )

        # ray propagation with the small angle approx
        alpha_x = self.alphax_foreground_MI - alpha_x_macro + self.alphax_background_MI
        alpha_y = self.alphay_foreground_MI - alpha_y_macro + self.alphay_background_MI
        x_source_grid = self.x0_MI + alpha_x * self.Tds
        y_source_grid = self.y0_MI + alpha_y * self.Tds

        npt.assert_allclose(x_source_true, x_source_grid)
        npt.assert_allclose(y_source_true, y_source_grid)

        lens_model_decoupled = LensModel(**self.kwargs_multiplane_model_MI)
        beta_x_new, beta_y_new = lens_model_decoupled.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_free
        )
        beta_x_true, beta_y_true = self.lens_model_true.ray_shooting(
            self.x_image, self.y_image, self.kwargs_lens_true
        )
        npt.assert_allclose(beta_x_new, beta_x_true)
        npt.assert_allclose(beta_y_new, beta_y_true)

    def test_setup_raytracing_lens_models(self):

        grid_size = 0.05
        grid_resolution = 0.001
        multiplane_lens_model_list, kwargs_multiplane_lens_model_list = (
            setup_raytracing_lensmodels(
                self.x_image,
                self.y_image,
                self.lens_model_true,
                self.kwargs_lens_true,
                self.index_lens_split,
                grid_size,
                grid_resolution,
            )
        )
        x_grid, y_grid, _, _ = setup_grids(grid_size, 0.0025)
        lens_model_decoupled_multiple_images = LensModel(
            **self.kwargs_multiplane_model_MI
        )

        for i in range(0, len(self.x_image)):
            beta_x, beta_y = multiplane_lens_model_list[i].ray_shooting(
                x_grid + self.x_image[i],
                y_grid + self.y_image[i],
                self.kwargs_lens_free,
            )
            beta_x_true, beta_y_true = self.lens_model_true.ray_shooting(
                x_grid + self.x_image[i],
                y_grid + self.y_image[i],
                self.kwargs_lens_true,
            )
            npt.assert_allclose(beta_x, beta_x_true, 5)
            npt.assert_allclose(beta_y, beta_y_true, 5)
            beta_x, beta_y = multiplane_lens_model_list[i].ray_shooting(
                self.x_image[i], self.y_image[i], self.kwargs_lens_free
            )
            beta_x_true, beta_y_true = (
                lens_model_decoupled_multiple_images.ray_shooting(
                    x_grid + self.x_image[i],
                    y_grid + self.y_image[i],
                    self.kwargs_lens_free,
                )
            )
            npt.assert_almost_equal(beta_x, beta_x_true)
            npt.assert_almost_equal(beta_y, beta_y_true)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

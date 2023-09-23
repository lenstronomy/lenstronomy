from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Util import param_util
from lenstronomy.Util import image_util
import pytest

import numpy as np
import numpy.testing as npt
import unittest

from lenstronomy.Analysis.multi_patch_reconstruction import (
    MultiPatchReconstruction,
    _update_frame_size,
)


class TestMultiPatchReconstruction(object):
    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel (Gaussian)
        exp_time = 100.0  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
        psf_type = "GAUSSIAN"  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # generate the coordinate grid and image properties
        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        kwargs_data["exposure_time"] = exp_time * np.ones_like(
            kwargs_data["image_data"]
        )
        data_class = ImageData(**kwargs_data)
        # generate the psf variables

        kwargs_psf = {"psf_type": psf_type, "pixel_size": deltaPix, "fwhm": fwhm}
        # kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)
        psf_class = PSF(**kwargs_psf)

        # lensing quantities
        kwargs_shear = {
            "gamma1": 0.02,
            "gamma2": -0.04,
        }  # shear values to the source plane
        kwargs_spemd = {
            "theta_E": 1.26,
            "gamma": 2.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "e1": -0.1,
            "e2": 0.05,
        }  # parameters of the deflector lens model

        # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list = ["EPL", "SHEAR"]
        kwargs_lens_true = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # choice of source type
        source_type = "SERSIC"  # 'SERSIC' or 'SHAPELETS'

        source_x = 0.0
        source_y = 0.05

        # Sersic parameters in the initial simulation
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_sersic_source = {
            "amp": 1000,
            "R_sersic": 0.05,
            "n_sersic": 1,
            "e1": e1,
            "e2": e2,
            "center_x": source_x,
            "center_y": source_y,
        }
        # kwargs_else = {'sourcePos_x': source_x, 'sourcePos_y': source_y, 'quasar_amp': 400., 'gamma1_foreground': 0.0, 'gamma2_foreground':-0.0}
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_source_true = [kwargs_sersic_source]
        source_model_class = LightModel(light_model_list=source_model_list)

        lensEquationSolver = LensEquationSolver(lens_model_class)
        x_image, y_image = lensEquationSolver.findBrightImage(
            source_x,
            source_y,
            kwargs_lens_true,
            numImages=4,
            min_distance=deltaPix,
            search_window=numPix * deltaPix,
        )
        mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens_true)

        kwargs_numerics = {"supersampling_factor": 1}

        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            kwargs_numerics=kwargs_numerics,
        )

        # generate image
        model = imageModel.image(kwargs_lens_true, kwargs_source_true)
        poisson = image_util.add_poisson(model, exp_time=exp_time)
        bkg = image_util.add_background(model, sigma_bkd=sigma_bkg)
        image_sim = model + bkg + poisson

        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim

        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
        }

        # make cutous and data instances of them
        x_pos, y_pos = data_class.map_coord2pix(x_image, y_image)
        ra_grid, dec_grid = data_class.pixel_coordinates

        multi_band_list = []
        for i in range(len(x_pos)):
            n_cut = 12
            x_c = int(x_pos[i])
            y_c = int(y_pos[i])
            image_cut = image_sim[
                int(y_c - n_cut) : int(y_c + n_cut), int(x_c - n_cut) : int(x_c + n_cut)
            ]
            exposure_map_cut = data_class.exposure_map[
                int(y_c - n_cut) : int(y_c + n_cut), int(x_c - n_cut) : int(x_c + n_cut)
            ]
            kwargs_data_i = {
                "background_rms": data_class.background_rms,
                "exposure_time": exposure_map_cut,
                "ra_at_xy_0": ra_grid[y_c - n_cut, x_c - n_cut],
                "dec_at_xy_0": dec_grid[y_c - n_cut, x_c - n_cut],
                "transform_pix2angle": data_class.transform_pix2angle,
                "image_data": image_cut,
            }
            multi_band_list.append([kwargs_data_i, kwargs_psf, kwargs_numerics])

        kwargs_params = {
            "kwargs_lens": kwargs_lens_true,
            "kwargs_source": kwargs_source_true,
        }
        self.multiPatch = MultiPatchReconstruction(
            multi_band_list,
            kwargs_model,
            kwargs_params,
            multi_band_type="joint-linear",
            kwargs_likelihood=None,
            verbose=True,
        )
        self.data_class = data_class
        self.model = model
        self.lens_model_class = lens_model_class
        self.kwargs_lens = kwargs_lens_true

        # test multi_patch with initial pixel grid
        kwargs_pixel_grid = {
            "nx": numPix,
            "ny": numPix,
            "transform_pix2angle": kwargs_data["transform_pix2angle"],
            "ra_at_xy_0": kwargs_data["ra_at_xy_0"],
            "dec_at_xy_0": kwargs_data["dec_at_xy_0"],
        }
        multiPatch = MultiPatchReconstruction(
            multi_band_list,
            kwargs_model,
            kwargs_params,
            multi_band_type="joint-linear",
            kwargs_pixel_grid=kwargs_pixel_grid,
        )

    def test_pixel_grid(self):
        pixel_grid = self.multiPatch.pixel_grid_joint
        nx, ny = pixel_grid.num_pixel_axes
        assert nx == 74
        assert ny == 67

    def test_image_joint(self):
        image_joint, model_joint, norm_residuals_joint = self.multiPatch.image_joint()

        # compute pixel shift from original
        pixel_grid = self.multiPatch.pixel_grid_joint
        nx, ny = pixel_grid.num_pixel_axes
        ra, dec = pixel_grid.radec_at_xy_0
        x0, y0 = self.data_class.map_coord2pix(ra, dec)
        # cutout original
        data = self.data_class.data
        data_cut = data[int(y0) : int(y0 + ny), int(x0) : int(x0 + nx)]
        model_cut = self.model[int(y0) : int(y0 + ny), int(x0) : int(x0 + nx)]
        # compare with original
        npt.assert_almost_equal(
            data_cut[image_joint > 0], image_joint[image_joint > 0], decimal=5
        )
        model_cut[model_joint == 0] = 0
        print(np.sum(model_cut), np.sum(model_joint), "test sum")
        # import matplotlib.pyplot as plt
        # plt.matshow((model_joint - model_cut))
        # plt.show()

        # plt.matshow(model_cut)
        # plt.show()
        # TODO make this test more precise (to do with narrower PSF convolution?)
        npt.assert_almost_equal(
            model_cut[model_joint > 0], model_joint[model_joint > 0], decimal=1
        )

    def test_lens_model_joint(self):
        (
            kappa_joint,
            magnification_joint,
            alpha_x_joint,
            alpha_y_joint,
        ) = self.multiPatch.lens_model_joint()

        # compute pixel shift from original
        pixel_grid = self.multiPatch.pixel_grid_joint
        nx, ny = pixel_grid.num_pixel_axes
        ra, dec = pixel_grid.radec_at_xy_0
        x0, y0 = self.data_class.map_coord2pix(ra, dec)
        # cutout original
        x_grid, y_grid = self.data_class.pixel_coordinates
        kappa = self.lens_model_class.kappa(x_grid, y_grid, self.kwargs_lens)
        kappa_cut = kappa[int(y0) : int(y0 + ny), int(x0) : int(x0 + nx)]
        # compare with original
        npt.assert_almost_equal(
            kappa_cut[kappa_joint > 0], kappa_joint[kappa_joint > 0], decimal=5
        )

        alpha_x, alpha_y = self.lens_model_class.alpha(x_grid, y_grid, self.kwargs_lens)
        alpha_x_cut = alpha_x[int(y0) : int(y0 + ny), int(x0) : int(x0 + nx)]
        # compare with original
        npt.assert_almost_equal(
            alpha_x_cut[alpha_x_joint > 0], alpha_x_joint[alpha_x_joint > 0], decimal=5
        )

    def test_source(self):
        source, coords = self.multiPatch.source(num_pix=50, delta_pix=0.01, center=None)
        nx, ny = np.shape(source)
        assert nx == 50

        source, coords = self.multiPatch.source(
            num_pix=50, delta_pix=0.01, center=[0, 0]
        )
        nx, ny = np.shape(source)
        assert nx == 50

    def test__update_frame_size(self):
        nx, ny = _update_frame_size(nx=10, ny=10, x_min=-5, y_min=2, nx_i=5, ny_i=5)
        assert nx == 15


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            MultiPatchReconstruction(
                multi_band_list=[],
                kwargs_model={},
                kwargs_params={},
                multi_band_type="multi-linear",
                kwargs_likelihood=None,
                verbose=True,
            )


if __name__ == "__main__":
    pytest.main()

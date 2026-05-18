import pytest
import numpy.testing as npt
import numpy as np

from lenstronomy.ImSim.MultiBand.single_band_multi_model import SingleBandMultiModel

from lenstronomy.Util import simulation_util, util
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestSingleBandMultiModel(object):
    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = simulation_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": deltaPix}
        psf_class = PSF(**kwargs_psf)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }
        kwargs_sie = {
            "theta_E": 1.0,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": -0.3,
        }
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }
        lens_model_list = ["SPEP", "SIE", "SHEAR"]
        kwargs_lens = [kwargs_spemd, kwargs_sie, kwargs_shear]
        kwargs_lens_imageModel = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=["SPEP", "SHEAR"])
        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 3,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        lens_light_model_list = ["SERSIC"]
        kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)

        # Point Source
        point_source_model_list = ["UNLENSED"]
        kwargs_ps = [{"ra_image": [0.4], "dec_image": [-0.2], "point_amp": [2]}]
        point_source_class = PointSource(point_source_type_list=point_source_model_list)

        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
            "compute_mode": "regular",
        }
        imageModel = ImageLinearFit(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class=point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.imageModel = imageModel
        image_sim = simulation_util.simulate_simple(
            imageModel,
            kwargs_lens_imageModel,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
        )

        data_class.update_data(image_sim)
        kwargs_data["image_data"] = image_sim
        self.multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]

        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": point_source_model_list,
            "fixed_magnification_list": [False],
            "index_lens_model_list": [[0, 2], [0, 1]],
            "index_lens_light_model_list": [[0], [0]],
            "index_source_light_model_list": [[0], [0]],
            "index_point_source_model_list": [[0], [0]],
            "point_source_frame_list": [[0], [0]],
        }

        self.kwargs_params = {
            "kwargs_lens": kwargs_lens,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }
        self.kwargs_params_imageModel = {
            "kwargs_lens": kwargs_lens_imageModel,
            "kwargs_source": kwargs_source,
            "kwargs_lens_light": kwargs_lens_light,
            "kwargs_ps": kwargs_ps,
        }

        self.single_band = SingleBandMultiModel(
            multi_band_list=self.multi_band_list,
            kwargs_model=self.kwargs_model,
            band_index=0,
            linear_solver=True,
        )
        self.single_band_no_linear = SingleBandMultiModel(
            multi_band_list=self.multi_band_list,
            kwargs_model=self.kwargs_model,
            band_index=0,
            linear_solver=False,
        )

    def test_likelihood_data_given_model(self):
        logl, _ = self.single_band.likelihood_data_given_model(**self.kwargs_params)
        logl_no_linear, param = self.single_band_no_linear.likelihood_data_given_model(
            **self.kwargs_params
        )
        npt.assert_almost_equal(logl / logl_no_linear, 1, decimal=4)

        # Tests the feature to deactivate/activate linear_solver
        logl_no_linear2, _ = self.single_band.likelihood_data_given_model(
            linear_solver=False, **self.kwargs_params
        )
        npt.assert_almost_equal(logl_no_linear2, logl_no_linear, decimal=8)

        logl2, _ = self.single_band_no_linear.likelihood_data_given_model(
            linear_solver=True, **self.kwargs_params
        )
        npt.assert_almost_equal(logl2, logl, decimal=8)

    def test_num_param_linear(self):
        num_linear = self.single_band.num_param_linear(**self.kwargs_params)
        assert num_linear == 3
        num_linear = self.single_band_no_linear.num_param_linear(**self.kwargs_params)
        assert num_linear == 0

    def test_image(self):
        image = self.single_band_no_linear.image(**self.kwargs_params)
        image_ = self.imageModel.image(**self.kwargs_params_imageModel)
        npt.assert_almost_equal(image, image_)

        image = self.single_band.image(**self.kwargs_params)
        image_ = self.imageModel.image(**self.kwargs_params_imageModel)
        npt.assert_almost_equal(image, image_)

    def test_source_surface_brightness(self):
        image = self.single_band.source_surface_brightness(
            kwargs_source=self.kwargs_params["kwargs_source"],
            kwargs_lens=self.kwargs_params["kwargs_lens"],
        )
        image_ = self.imageModel.source_surface_brightness(
            kwargs_source=self.kwargs_params_imageModel["kwargs_source"],
            kwargs_lens=self.kwargs_params_imageModel["kwargs_lens"],
        )
        npt.assert_almost_equal(image, image_)

    def test_lens_surface_brightness(self):
        image = self.single_band.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_params["kwargs_lens_light"]
        )
        image_ = self.imageModel.lens_surface_brightness(
            kwargs_lens_light=self.kwargs_params_imageModel["kwargs_lens_light"]
        )
        npt.assert_almost_equal(image, image_)

    def test_point_source(self):
        image = self.single_band.point_source(
            kwargs_lens=self.kwargs_params["kwargs_lens"],
            kwargs_ps=self.kwargs_params["kwargs_ps"],
        )
        image_ = self.imageModel.point_source(
            kwargs_lens=self.kwargs_params_imageModel["kwargs_lens"],
            kwargs_ps=self.kwargs_params_imageModel["kwargs_ps"],
        )
        npt.assert_almost_equal(image, image_)

    def test_update_linear_kwargs(self):
        num = self.single_band.num_param_linear(
            self.kwargs_params["kwargs_lens"],
            self.kwargs_params["kwargs_source"],
            self.kwargs_params["kwargs_lens_light"],
            self.kwargs_params["kwargs_ps"],
        )
        param = np.ones(num) * 10
        (
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps,
        ) = self.single_band.update_linear_kwargs(
            param,
            kwargs_lens=self.kwargs_params["kwargs_lens"],
            kwargs_source=self.kwargs_params["kwargs_ps"],
            kwargs_lens_light=self.kwargs_params["kwargs_lens_light"],
            kwargs_ps=self.kwargs_params["kwargs_ps"],
        )
        assert kwargs_source[0]["amp"] == 10

    def test_extinction_map(self):
        extinction_map = self.single_band.extinction_map(
            kwargs_extinction=None, kwargs_special=None
        )
        npt.assert_almost_equal(extinction_map, 1)

    def test_error_map_source(self):
        x_grid, y_grid = util.make_grid(numPix=10, deltapix=0.1)
        error = self.single_band_no_linear.error_map_source(
            [None], x_grid, y_grid, None
        )
        npt.assert_array_equal(error, np.zeros(100))

        kwargs_source = self.kwargs_params["kwargs_source"]
        error = self.single_band.error_map_source(kwargs_source, x_grid, y_grid, None)
        error2 = self.imageModel.error_map_source(kwargs_source, x_grid, y_grid, None)
        npt.assert_array_almost_equal(error, error2, decimal=8)

    def test_linear_param_from_kwargs(self):
        kwargs_source = self.kwargs_params["kwargs_source"]
        kwargs_lens_light = self.kwargs_params["kwargs_lens_light"]
        kwargs_ps = self.kwargs_params["kwargs_ps"]

        linear_params = self.single_band.linear_param_from_kwargs(
            kwargs_source, kwargs_lens_light, kwargs_ps
        )
        linear_params2 = self.imageModel.linear_param_from_kwargs(
            kwargs_source, kwargs_lens_light, kwargs_ps
        )
        npt.assert_array_almost_equal(linear_params, linear_params2, decimal=8)


if __name__ == "__main__":
    pytest.main()

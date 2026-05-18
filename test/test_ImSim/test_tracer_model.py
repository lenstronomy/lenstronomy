__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.differential_extinction import DifferentialExtinction
from lenstronomy.Util import util

from lenstronomy.ImSim.tracer_model import TracerModelSource


class TestTracerModel(object):
    """Test TracerModel class."""

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg, inverse=True
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_sis = {"theta_E": 1.0, "center_x": 0, "center_y": 0}
        lens_model_list = ["SIS"]
        self.kwargs_lens = [kwargs_sis]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.4,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        source_light_model_list = ["SERSIC"]
        self.kwargs_source_light = [kwargs_sersic]
        source_model_class = LightModel(light_model_list=source_light_model_list)

        # Tracer model
        tracer_model = ["LINEAR"]
        self.kwargs_tracer = [{"amp": 1, "k": 2, "center_x": 0, "center_y": 0}]
        tracer_source_class = LightModel(light_model_list=tracer_model)
        kwargs_numerics = {
            "supersampling_factor": 2,
            "supersampling_convolution": False,
        }
        self.linearTracerModel = TracerModelSource(
            data_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            tracer_source_class=tracer_source_class,
            kwargs_numerics=kwargs_numerics,
            tracer_type="LINEAR",
        )
        self.logTracerModel = TracerModelSource(
            data_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            tracer_source_class=tracer_source_class,
            kwargs_numerics=kwargs_numerics,
            tracer_type="LOG",
        )

    def test_linear_tracer_model(self):
        tracer_model = self.linearTracerModel.tracer_model(
            self.kwargs_tracer,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source_light,
        )

        light_unconvolved = self.linearTracerModel.source_surface_brightness(
            kwargs_source=self.kwargs_source_light,
            kwargs_lens=self.kwargs_lens,
            unconvolved=True,
        )

        light_convolved = self.linearTracerModel.source_surface_brightness(
            kwargs_source=self.kwargs_source_light,
            kwargs_lens=self.kwargs_lens,
            unconvolved=False,
        )

        source_light_num = (
            self.linearTracerModel._source_surface_brightness_analytical_numerics(
                self.kwargs_source_light, self.kwargs_lens, de_lensed=False
            )
        )
        tracer = self.linearTracerModel._tracer_model_source(
            self.kwargs_tracer, self.kwargs_lens, de_lensed=False
        )
        tracer_brightness_conv = self.linearTracerModel.ImageNumerics.re_size_convolve(
            tracer * source_light_num, unconvolved=False
        )
        tracer_model_2 = tracer_brightness_conv / light_convolved
        npt.assert_almost_equal(tracer_model_2, tracer_model, decimal=5)

    def test_log_tracer_model(self):
        """Convolve a tracer with log units with a surface brightness profile in two ways:
        (1) using the tracer_type="LOG" argument
        (2) converting units to linear ones, using tracer_type="LINEAR", and
        converting back to log units."""
        tracer = self.logTracerModel._tracer_model_source(
            self.kwargs_tracer, self.kwargs_lens, de_lensed=False
        )

        source_light_num = (
            self.logTracerModel._source_surface_brightness_analytical_numerics(
                self.kwargs_source_light, self.kwargs_lens, de_lensed=False
            )
        )

        light_unconvolved = self.linearTracerModel.source_surface_brightness(
            kwargs_source=self.kwargs_source_light,
            kwargs_lens=self.kwargs_lens,
            unconvolved=True,
        )

        light_convolved = self.linearTracerModel.source_surface_brightness(
            kwargs_source=self.kwargs_source_light,
            kwargs_lens=self.kwargs_lens,
            unconvolved=False,
        )

        # Uses tracer_type='LOG'
        tracer_model_1 = self.logTracerModel.tracer_model(
            self.kwargs_tracer,
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source_light,
        )

        # Converts to linear units; convolves; converts back.
        linearized_tracer = np.power(10, tracer)
        linearized_tracer_brightness_conv = (
            self.logTracerModel.ImageNumerics.re_size_convolve(
                linearized_tracer * source_light_num, unconvolved=False
            )
        )
        tracer_model_2 = np.log10(linearized_tracer_brightness_conv / light_convolved)

        npt.assert_almost_equal(tracer_model_1, tracer_model_2, decimal=5)

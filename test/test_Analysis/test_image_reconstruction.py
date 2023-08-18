__author__ = "sibirrer"

import pytest
import numpy as np
import numpy.testing as npt
import unittest
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Plots.model_plot import check_solver_error
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Analysis.image_reconstruction import MultiBandImageReconstruction


class TestMultiBandImageReconstruction(object):
    """
    test the fitting sequences
    """

    def setup_method(self):
        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.5  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        self.kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**self.kwargs_data)
        self.kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        psf_class = PSF(**self.kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {
            "gamma1": 0.01,
            "gamma2": 0.01,
        }  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        e1, e2 = param_util.phi_q2_ellipticity(0.2, 0.8)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SPEP", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        self.LensModel = lens_model_class
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [
            {"ra_source": 0.0, "dec_source": 0.0, "source_amp": 1.0}
        ]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = ["SOURCE_POSITION"]
        point_source_class = PointSource(
            point_source_type_list=point_source_list, fixed_magnification_list=[True]
        )
        kwargs_numerics = {"supersampling_factor": 1}
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            no_noise=True,
        )

        data_class.update_data(image_sim)
        self.kwargs_data["image_data"] = image_sim
        self.kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "point_source_model_list": point_source_list,
            "fixed_magnification_list": [False],
        }
        self.kwargs_numerics = kwargs_numerics
        self.data_class = ImageData(**self.kwargs_data)
        self.kwargs_params = {
            "kwargs_lens": self.kwargs_lens,
            "kwargs_source": self.kwargs_source,
            "kwargs_lens_light": self.kwargs_lens_light,
            "kwargs_ps": self.kwargs_ps,
        }

    def test_band_setup(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        multi_band = MultiBandImageReconstruction(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            multi_band_type="single-band",
        )

        multi_band = MultiBandImageReconstruction(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            multi_band_type="joint-linear",
        )

        image_model, kwargs_params = multi_band.band_setup(band_index=0)
        model = image_model.image(**kwargs_params)
        npt.assert_almost_equal(model, self.kwargs_data["image_data"], decimal=5)
        npt.assert_almost_equal(model, multi_band.model_band_list[0].model, decimal=5)
        npt.assert_almost_equal(
            multi_band.model_band_list[0].norm_residuals, 0, decimal=5
        )

    def test_bands_compute(self):
        multi_band_list = [
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        ] * 2
        multi_band = MultiBandImageReconstruction(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            kwargs_likelihood={"bands_compute": [False, True]},
            multi_band_type="multi-linear",
        )
        image_model, kwargs_params = multi_band.band_setup(band_index=1)
        model = image_model.image(**kwargs_params)
        npt.assert_almost_equal(model, self.kwargs_data["image_data"], decimal=5)

    def test_not_verbose(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        multi_band = MultiBandImageReconstruction(
            multi_band_list, self.kwargs_model, self.kwargs_params, verbose=False
        )

    def test_check_solver_error(self):
        bool = check_solver_error(image=np.array([0, 0]))
        assert bool

        bool = check_solver_error(image=np.array([0, 0.1]))
        assert bool == 0


class TestRaises(unittest.TestCase):
    def test_no_band(self):
        """
        test raise statements if band is not evaluated

        """
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 10  # cutout pixel size
        deltaPix = 0.5  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        kwargs_data["image_data"] = np.ones((numPix, numPix))
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        kwargs_numerics = {}
        multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
        multi_band = MultiBandImageReconstruction(
            multi_band_list,
            {},
            {},
            multi_band_type="single-band",
            kwargs_likelihood={"bands_compute": [False]},
        )
        with self.assertRaises(ValueError):
            multi_band.band_setup(band_index=0)


if __name__ == "__main__":
    pytest.main()

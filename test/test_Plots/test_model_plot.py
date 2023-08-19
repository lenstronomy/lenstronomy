__author__ = "sibirrer"

import pytest
import numpy as np
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots.model_plot import check_solver_error
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest


class TestOutputPlots(object):
    """Test the fitting sequences."""

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
        kwargs_psf_gaussian = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "truncation": 5,
            "pixel_size": deltaPix,
        }
        psf_gaussian = PSF(**kwargs_psf_gaussian)
        self.kwargs_psf = {
            "psf_type": "PIXEL",
            "kernel_point_source": psf_gaussian.kernel_point_source,
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
        lens_model_class = LensModel(
            lens_model_list=lens_model_list,
            multi_plane=True,
            lens_redshift_list=[0.5, 0.5],
            z_source=2.0,
        )
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
        self.kwargs_model_multiplane = {
            "lens_model_list": lens_model_list,
            "lens_redshift_list": [0.5, 0.5],
            "multi_plane": True,
            "z_source": 2.0,
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

    def test_lensModelPlot(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            multi_band_type="single-band",
        )

        multi_band_list_multiplane = [
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]
        ]
        lensPlot_multiplane = ModelPlot(
            multi_band_list_multiplane,
            self.kwargs_model_multiplane,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            multi_band_type="single-band",
        )

        lensPlot.plot_main(with_caustics=True)
        plt.close()
        cmap = plt.get_cmap("gist_heat")

        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string=cmap,
        )

        lensPlot.plot_separate()
        plt.close()
        lensPlot.plot_subtract_from_data_all()
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.deflection_plot(ax=ax, with_caustics=True, axis=1)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.subtract_from_data_plot(ax=ax)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.deflection_plot(ax=ax, with_caustics=True, axis=0)
        plt.close()

        numPix = 100
        deltaPix_source = 0.01
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.error_map_source_plot(
            ax=ax, numPix=numPix, deltaPix_source=deltaPix_source, with_caustics=True
        )
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.absolute_residual_plot(ax=ax)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.plot_extinction_map(ax=ax)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        kwargs_plot = {"index_macromodel": [0]}
        lensPlot.substructure_plot(ax=ax, **kwargs_plot)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        kwargs_plot = {"index_macromodel": [0], "with_critical_curves": True}
        lensPlot.substructure_plot(ax=ax, **kwargs_plot)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        kwargs_plot = {"index_macromodel": [0], "with_critical_curves": True}
        lensPlot_multiplane.substructure_plot(ax=ax, **kwargs_plot)
        plt.close()

    def test_source_plot(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            fast_caustic=False,
        )

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.source_plot(
            ax=ax,
            numPix=10,
            deltaPix_source=0.1,
            v_min=None,
            v_max=None,
            with_caustics=True,
            caustic_color="yellow",
            fsize=15,
            plot_scale="linear",
        )
        plt.close()

    def test_source(self):
        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
        )
        source, coords_source = lensPlot.source(
            band_index=0, numPix=10, deltaPix=0.1, image_orientation=True
        )
        assert len(source) == 10

        source, coords_source = lensPlot.source(
            band_index=0, numPix=10, deltaPix=0.1, image_orientation=False
        )
        assert len(source) == 10

        source, coords_source = lensPlot.source(
            band_index=0, numPix=10, deltaPix=0.1, center=[0, 0]
        )
        assert len(source) == 10

    def test_joint_linear(self):
        multi_band_list = [
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics],
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics],
        ]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            multi_band_type="joint-linear",
            bands_compute=[True, False],
        )

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.data_plot(
            ax=ax,
            numPix=10,
            deltaPix_source=0.1,
            v_min=None,
            v_max=None,
            with_caustics=False,
            caustic_color="yellow",
            fsize=15,
            plot_scale="linear",
        )
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.model_plot(
            ax=ax,
            numPix=10,
            deltaPix_source=0.1,
            v_min=None,
            v_max=None,
            with_caustics=False,
            caustic_color="yellow",
            fsize=15,
            plot_scale="linear",
        )
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.convergence_plot(
            ax=ax,
            numPix=10,
            deltaPix_source=0.1,
            v_min=None,
            v_max=None,
            with_caustics=False,
            caustic_color="yellow",
            fsize=15,
            plot_scale="linear",
        )
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.normalized_residual_plot(ax=ax)
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.magnification_plot(ax=ax)
        plt.close()
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = lensPlot.decomposition_plot(ax=ax)
        plt.close()

    def test_reconstruction_all_bands(self):
        multi_band_list = [
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics],
            [self.kwargs_data, self.kwargs_psf, self.kwargs_numerics],
        ]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            multi_band_type="joint-linear",
            bands_compute=[True, True],
        )
        f, axes = lensPlot.reconstruction_all_bands()
        assert len(axes) == 2
        assert len(axes[0]) == 3
        plt.close()

        multi_band_list = [[self.kwargs_data, self.kwargs_psf, self.kwargs_numerics]]
        lensPlot = ModelPlot(
            multi_band_list,
            self.kwargs_model,
            self.kwargs_params,
            arrow_size=0.02,
            cmap_string="gist_heat",
            multi_band_type="joint-linear",
            bands_compute=[True],
        )
        f, axes = lensPlot.reconstruction_all_bands()
        assert len(axes) == 1
        assert len(axes[0]) == 3
        plt.close()

    def test_check_solver_error(self):
        bool = check_solver_error(image=np.array([0, 0]))
        assert bool

        bool = check_solver_error(image=np.array([0, 0.1]))
        assert bool == 0

    def test_no_linear_solver(self):
        kwargs_data = sim_util.data_configure_simple(
            numPix=10, deltaPix=1, background_rms=1, exposure_time=1
        )
        # kwargs_data['image_data'] = np.zeros((10, 10))
        kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
        kwargs_params = {
            "kwargs_lens": [],
            "kwargs_source": [{"amp": 2, "sigma": 1, "center_x": 0, "center_y": 0}],
            "kwargs_ps": [],
            "kwargs_lens_light": [],
        }
        lensPlot = ModelPlot(
            multi_band_list=[[kwargs_data, {"psf_type": "NONE"}, {}]],
            kwargs_model=kwargs_model,
            kwargs_params=kwargs_params,
            bands_compute=[True],
            arrow_size=0.02,
            cmap_string="gist_heat",
            linear_solver=False,
        )
        lensPlot.plot_main(with_caustics=True)
        plt.close()
        assert kwargs_params["kwargs_source"][0]["amp"] == 2


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            kwargs_data = sim_util.data_configure_simple(
                numPix=10, deltaPix=1, background_rms=1
            )
            # kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
            kwargs_params = {
                "kwargs_lens": [],
                "kwargs_source": [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}],
                "kwargs_ps": [],
                "kwargs_lens_light": [],
            }
            lensPlot = ModelPlot(
                multi_band_list=[[kwargs_data, {"psf_type": "NONE"}, {}]],
                kwargs_model=kwargs_model,
                kwargs_params=kwargs_params,
                arrow_size=0.02,
                cmap_string="gist_heat",
            )
        with self.assertRaises(ValueError):
            kwargs_data = sim_util.data_configure_simple(
                numPix=10, deltaPix=1, background_rms=1
            )
            # kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
            kwargs_params = {
                "kwargs_lens": [],
                "kwargs_source": [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}],
                "kwargs_ps": [],
                "kwargs_lens_light": [],
            }
            lensPlot = ModelPlot(
                multi_band_list=[[kwargs_data, {}, {}]],
                kwargs_model=kwargs_model,
                kwargs_params=kwargs_params,
                arrow_size=0.02,
                cmap_string="gist_heat",
            )
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax = lensPlot.source_plot(
                ax=ax,
                numPix=10,
                deltaPix_source=0.1,
                v_min=None,
                v_max=None,
                with_caustics=False,
                caustic_color="yellow",
                fsize=15,
                plot_scale="bad",
            )
            plt.close()
        with self.assertRaises(ValueError):
            kwargs_data = sim_util.data_configure_simple(
                numPix=10, deltaPix=1, background_rms=1
            )
            # kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
            kwargs_params = {
                "kwargs_lens": [],
                "kwargs_source": [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}],
                "kwargs_ps": [],
                "kwargs_lens_light": [],
            }
            lensPlot = ModelPlot(
                multi_band_list=[[kwargs_data, {"psf_type": "NONE"}, {}]],
                kwargs_model=kwargs_model,
                kwargs_params=kwargs_params,
                bands_compute=[False],
                arrow_size=0.02,
                cmap_string="gist_heat",
            )
            lensPlot._select_band(band_index=0)

        with self.assertRaises(ValueError):
            kwargs_data = sim_util.data_configure_simple(
                numPix=10, deltaPix=1, background_rms=1, exposure_time=1
            )
            # kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
            kwargs_params = {
                "kwargs_lens": [],
                "kwargs_source": [{"amp": 1, "sigma": 1, "center_x": 0, "center_y": 0}],
                "kwargs_ps": [],
                "kwargs_lens_light": [],
            }
            lensPlot = ModelPlot(
                multi_band_list=[[kwargs_data, {"psf_type": "NONE"}, {}]],
                kwargs_model=kwargs_model,
                kwargs_params=kwargs_params,
                bands_compute=[True],
                arrow_size=0.02,
                cmap_string="gist_heat",
            )

            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax = lensPlot.source_plot(
                ax=ax,
                numPix=10,
                deltaPix_source=0.1,
                v_min=None,
                v_max=None,
                with_caustics=False,
                caustic_color="yellow",
                fsize=15,
                plot_scale="wrong",
            )
            plt.close()
        with self.assertRaises(ValueError):
            # test whether linear_solver=False returns raise when having two bands
            kwargs_data = sim_util.data_configure_simple(
                numPix=10, deltaPix=1, background_rms=1, exposure_time=1
            )
            # kwargs_data['image_data'] = np.zeros((10, 10))
            kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
            kwargs_params = {
                "kwargs_lens": [],
                "kwargs_source": [{"amp": 2, "sigma": 1, "center_x": 0, "center_y": 0}],
                "kwargs_ps": [],
                "kwargs_lens_light": [],
            }
            lensPlot = ModelPlot(
                multi_band_list=[
                    [kwargs_data, {"psf_type": "NONE"}, {}],
                    [kwargs_data, {"psf_type": "NONE"}, {}],
                ],
                kwargs_model=kwargs_model,
                kwargs_params=kwargs_params,
                bands_compute=[True],
                arrow_size=0.02,
                cmap_string="gist_heat",
                linear_solver=False,
            )


def test_interferometry_natwt_Model_Plot_linear_solver():
    # Test no errors are raised in the Model Plot linear solver for 'interferometry_natwt' likelihood function.
    try:
        kwargs_data = sim_util.data_configure_simple(
            numPix=10, deltaPix=1, background_rms=1, exposure_time=1
        )
        kwargs_data["likelihood_method"] = "interferometry_natwt"
        kwargs_model = {"source_light_model_list": ["GAUSSIAN"]}
        kwargs_params = {
            "kwargs_lens": [],
            "kwargs_source": [{"amp": 2, "sigma": 1, "center_x": 0, "center_y": 0}],
            "kwargs_ps": [],
            "kwargs_lens_light": [],
        }
        lensPlot = ModelPlot(
            multi_band_list=[[kwargs_data, {"psf_type": "NONE"}, {}]],
            kwargs_model=kwargs_model,
            kwargs_params=kwargs_params,
            bands_compute=[True],
            arrow_size=0.02,
            cmap_string="gist_heat",
            linear_solver=True,
        )
    except:
        pytest.fail(
            "Errors are raised in the Model Plot linear solver for the 'interferometric_natwt' likelihood method, which is not expected."
        )


if __name__ == "__main__":
    pytest.main()

__author__ = "sibirrer"

import pytest
import numpy as np
import numpy.testing as npt
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Sampling.likelihood import LikelihoodModule
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Data.kinematic_bin_2D import KinBin
from lenstronomy.Sampling.Likelihoods import kinematic_NN_call
import lenstronomy.Util.kernel_util as kernel_util


class TestLikelihoodModule(object):
    """Test the fitting sequences."""

    def setup_method(self):
        np.random.seed(42)

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 50  # cutout pixel size
        deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        kwargs_model = {
            "lens_model_list": ["SPEP"],
            "lens_light_model_list": ["SERSIC"],
            "source_light_model_list": ["SERSIC"],
            "point_source_model_list": ["SOURCE_POSITION"],
            "fixed_magnification_list": [True],
        }

        # PSF specification
        kwargs_band = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_band)
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": deltaPix}
        psf_class = PSF(**kwargs_psf)
        print(np.shape(psf_class.kernel_point_source), "test kernel shape -")
        kwargs_spep = {
            "theta_E": 1.0,
            "gamma": 1.95,
            "center_x": 0,
            "center_y": 0,
            "e1": 0.1,
            "e2": 0.1,
        }

        self.kwargs_lens = [kwargs_spep]
        kwargs_sersic = {
            "amp": 1 / 0.05**2.0,
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
        }

        self.kwargs_lens_light = [kwargs_sersic]
        self.kwargs_source = [kwargs_sersic_ellipse]
        self.kwargs_ps = [
            {"ra_source": 0.05, "dec_source": 0.02, "source_amp": 1.0}
        ]  # quasar point source position in the source plane and intrinsic brightness
        self.kwargs_cosmo = {"D_dt": 1000}
        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }
        (
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
        ) = class_creator.create_class_instances(**kwargs_model)
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            extinction_class,
            kwargs_numerics=kwargs_numerics,
        )
        image_sim = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
        )
        ra_pos, dec_pos = imageModel.PointSource.image_position(
            kwargs_ps=self.kwargs_ps, kwargs_lens=self.kwargs_lens
        )

        data_class.update_data(image_sim)
        kwargs_band["image_data"] = image_sim
        self.data_class = data_class
        self.psf_class = psf_class

        self.kwargs_model = kwargs_model
        self.kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        kwargs_constraints = {
            "num_point_source_list": [4],
            "solver_type": "NONE",  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
            "Ddt_sampling": True,
        }

        def condition_definition(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light,
            kwargs_ps=None,
            kwargs_special=None,
            kwargs_extinction=None,
            **kwargs,
        ):
            logL = 0
            if kwargs_lens_light[0]["R_sersic"] > kwargs_source[0]["R_sersic"]:
                logL -= 10**15
            return logL

        kwargs_likelihood = {
            "force_no_add_image": True,
            "source_marg": True,
            "astrometric_likelihood": True,
            "image_position_uncertainty": 0.004,
            "source_position_tolerance": 0.001,
            "source_position_likelihood": True,
            "source_position_sigma": 0.001,
            "flux_ratio_likelihood": True,
            "prior_lens": [[0, "theta_E", 1, 0.1]],
            "custom_logL_addition": condition_definition,
            "image_position_likelihood": True,
        }
        self.kwargs_data = {
            "multi_band_list": [[kwargs_band, kwargs_psf, kwargs_numerics]],
            "multi_band_type": "single-band",
            "time_delays_measured": np.ones(3),
            "time_delays_uncertainties": np.ones(3),
            "flux_ratios": np.ones(3),
            "flux_ratio_errors": np.ones(3),
            "ra_image_list": ra_pos,
            "dec_image_list": dec_pos,
        }
        self.param_class = Param(self.kwargs_model, **kwargs_constraints)
        self.imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        self.Likelihood = LikelihoodModule(
            kwargs_data_joint=self.kwargs_data,
            kwargs_model=kwargs_model,
            param_class=self.param_class,
            **kwargs_likelihood,
        )
        self.kwargs_band = kwargs_band
        self.kwargs_psf = kwargs_psf
        self.numPix = numPix

    def test_logL(self):
        args = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_cosmo,
        )

        logL = self.Likelihood.logL(args, verbose=True)
        num_data_evaluate = self.Likelihood.num_data
        npt.assert_almost_equal(logL / num_data_evaluate, -1 / 2.0, decimal=1)

    def test_time_delay_likelihood(self):
        kwargs_likelihood = {
            "time_delay_likelihood": True,
        }
        likelihood = LikelihoodModule(
            kwargs_data_joint=self.kwargs_data,
            kwargs_model=self.kwargs_model,
            param_class=self.param_class,
            **kwargs_likelihood,
        )
        args = self.param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
            kwargs_special=self.kwargs_cosmo,
        )

        logL = likelihood.logL(args, verbose=True)
        npt.assert_almost_equal(logL, -1328.821179288249, decimal=-1)

    def test_kin_likelihood(self):
        self.kinematic_NN = kinematic_NN_call.KinematicNN().SKiNN_installed
        if self.kinematic_NN:
            # since this doesn't work for the SPEP model, test it with a different lens model:
            kwargs_eplqphi = {
                "theta_E": 1.0,
                "gamma": 1.95,
                "center_x": 0,
                "center_y": 0,
                "q": 0.7,
                "phi": 0.7,
            }
            kwargs_lens = [kwargs_eplqphi]
            kwargs_sersic_ellipse_qphi = {
                "amp": 1.0,
                "R_sersic": 0.51,
                "n_sersic": 2.5,
                "center_x": 0,
                "center_y": 0,
                "q": 0.99,
                "phi": 0.7,
            }
            kwargs_lens_light = [kwargs_sersic_ellipse_qphi]
            kwargs_model = {
                "lens_model_list": ["EPL_Q_PHI"],
                "lens_light_model_list": ["SERSIC_ELLIPSE_Q_PHI"],
                "source_light_model_list": ["SERSIC"],
                "point_source_model_list": ["SOURCE_POSITION"],
                "fixed_magnification_list": [True],
                "z_lens": 1,
            }
            kwargs_constraints = {
                "num_point_source_list": [4],
                "solver_type": "NONE",  # 'PROFILE', 'PROFILE_SHEAR', 'ELLIPSE', 'CENTER'
                "Ddt_sampling": True,
            }
            kwargs_likelihood = {
                "kinematic_2d_likelihood": False,
                "kin_lens_idx": 0,
                "kin_lens_light_idx": 0,
                "time_delay_likelihood": False,
            }
            param_class = Param(kwargs_model, **kwargs_constraints)
            Likelihood = LikelihoodModule(
                kwargs_data_joint=self.kwargs_data,
                kwargs_model=kwargs_model,
                param_class=param_class,
                **kwargs_likelihood,
            )
            kwargs_cosmo = {
                "D_dt": 2000,
                "D_d": 1000,
                "b_ani": 0.2,
                "incli": np.pi / 2 - 0.01,
            }
            args = param_class.kwargs2args(
                kwargs_lens=kwargs_lens,
                kwargs_source=self.kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=self.kwargs_ps,
                kwargs_special=kwargs_cosmo,
            )
            logL_nokin = Likelihood.logL(args, verbose=True)
            # logL should be same order as SPEP value, buthas different r_sersic so it will be worse
            npt.assert_allclose(logL_nokin, -1328.821179288249, rtol=1)

            # Now add kinematic likelihood
            # for simplicity, set kin image data to same as light data
            numPix = 50  # cutout pixel size
            deltaPix = 0.1  # pixel size in arcsec (area per pixel = deltaPix**2)

            binmap = np.zeros_like(
                self.kwargs_band["image_data"]
            )  # one single bin across whole image
            binned_dummy_data = np.array([200])
            delta_pix_kin = deltaPix
            npix_kin = numPix

            kwargs_kin = {
                "bin_data": binned_dummy_data,
                "bin_cov": np.diag((binned_dummy_data * 0.05) ** 2),  # 5% error
                "bin_mask": binmap,
                "ra_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
                "dec_at_xy_0": -(npix_kin - 1) / 2.0 * delta_pix_kin,
                "transform_pix2angle": np.array([[1, 0], [0, 1]]) * delta_pix_kin,
            }

            kinkernel_point_source = kernel_util.kernel_gaussian(
                num_pix=9, delta_pix=0.2, fwhm=1.0
            )
            kwargs_pixelkin = {
                "psf_type": "PIXEL",
                "kernel_point_source": kinkernel_point_source,
            }
            kinPSF = PSF(**kwargs_pixelkin)
            _KinBin = KinBin(psf_class=kinPSF, **kwargs_kin)
            kwargs_data_kin = (
                self.kwargs_data.copy()
            )  # add kinematics data to kwargs_data
            kwargs_data_kin["kinematic_data"] = _KinBin

            # confirm that full likelihood is now lens+kin
            kwargs_likelihood = {
                "kinematic_2d_likelihood": True,
                "kin_lens_idx": 0,
                "kin_lens_light_idx": 0,
                "time_delay_likelihood": False,
            }

            kwargs_constraints["kinematic_sampling"] = True
            param_class = Param(kwargs_model, **kwargs_constraints)
            Likelihood = LikelihoodModule(
                kwargs_data_joint=kwargs_data_kin,
                kwargs_model=kwargs_model,
                param_class=param_class,
                **kwargs_likelihood,
            )
            args = param_class.kwargs2args(
                kwargs_lens=kwargs_lens,
                kwargs_source=self.kwargs_source,
                kwargs_lens_light=kwargs_lens_light,
                kwargs_ps=self.kwargs_ps,
                kwargs_special=kwargs_cosmo,
            )

            logL = Likelihood.logL(args, verbose=True)
            # With only one bin, the new logL should be worse than without kin by
            # half the mean chi2 averaged over the whole image
            image_averaged_chi2 = (
                (np.mean(Likelihood.kinematic_2D_likelihood.vrms) - binned_dummy_data)
                ** 2
                / kwargs_kin["bin_cov"]
            ).flatten()
            npt.assert_almost_equal(logL_nokin - logL, image_averaged_chi2 / 2)

    def test_check_bounds(self):
        penalty, bound_hit = self.Likelihood.check_bounds(
            args=[0, 1], lowerLimit=[1, 0], upperLimit=[2, 2], verbose=True
        )
        assert bound_hit

    def test_pixelbased_modelling(self):
        ss_source = 2
        numPix_source = self.numPix * ss_source
        n_scales = 3
        kwargs_pixelbased = {
            "source_interpolation": "nearest",
            "supersampling_factor_source": ss_source,  # supersampling of pixelated source grid
            # following choices are to minimize pixel solver runtime (not to get accurate reconstruction!)
            "threshold_decrease_type": "none",
            "num_iter_source": 2,
            "num_iter_lens": 2,
            "num_iter_global": 2,
            "num_iter_weights": 2,
        }
        kwargs_likelihood = {
            "image_likelihood": True,
            "kwargs_pixelbased": kwargs_pixelbased,
            "check_positive_flux": True,  # effectively not applied, activated for code coverage purposes
        }
        kernel = PSF(**self.kwargs_psf).kernel_point_source
        kwargs_psf = {"psf_type": "PIXEL", "kernel_point_source": kernel}
        kwargs_numerics = {"supersampling_factor": 1}
        kwargs_data = {
            "multi_band_list": [[self.kwargs_band, kwargs_psf, kwargs_numerics]]
        }
        kwargs_model = {
            "lens_model_list": ["SPEP"],
            "lens_light_model_list": ["SLIT_STARLETS"],
            "source_light_model_list": ["SLIT_STARLETS"],
        }

        kwargs_fixed_source = [
            {
                "n_scales": n_scales,
                "n_pixels": numPix_source**2,
                "scale": 1,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        kwargs_fixed_lens_light = [
            {
                "n_scales": n_scales,
                "n_pixels": self.numPix**2,
                "scale": 1,
                "center_x": 0,
                "center_y": 0,
            }
        ]
        kwargs_constraints = {"source_grid_offset": True}
        param_class = Param(
            kwargs_model,
            kwargs_fixed_source=kwargs_fixed_source,
            kwargs_fixed_lens_light=kwargs_fixed_lens_light,
            **kwargs_constraints,
        )

        likelihood = LikelihoodModule(
            kwargs_data_joint=kwargs_data,
            kwargs_model=kwargs_model,
            param_class=param_class,
            **kwargs_likelihood,
        )

        kwargs_source = [{"amp": np.ones(n_scales * numPix_source**2)}]
        kwargs_lens_light = [{"amp": np.ones(n_scales * self.numPix**2)}]
        kwargs_special = {"delta_x_source_grid": 0, "delta_y_source_grid": 0}
        args = param_class.kwargs2args(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_special=kwargs_special,
        )

        logL = likelihood.logL(args, verbose=True)
        num_data_evaluate = likelihood.num_data
        npt.assert_almost_equal(logL / num_data_evaluate, -1 / 2.0, decimal=1)

    def test_multi_source_redshift_likelihood(self):
        """This function tests the likelihood with multiple point sources at different
        source redshift and the recovery of log likelihood=0 when providing the exact
        solution."""
        # lens properties
        z_lens = 0.5
        # source properties
        z_source_convention = 1.5
        num_sources = 4
        z_sources = np.linspace(start=z_lens + 0.5, stop=z_lens + 2, num=num_sources)
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=100, Om0=0.3, Ob0=0.0)
        point_source_model_list = ["LENSED_POSITION"] * num_sources

        x_source, y_source = 0.1, 0.2
        # chose a lens model
        lens_model_list = ["SIE"]
        kwargs_lens = [
            {"theta_E": 1, "e1": 0.2, "e2": -0.2, "center_x": 0, "center_y": 0}
        ]

        from lenstronomy.Cosmo.lens_cosmo import LensCosmo

        lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source_convention, cosmo=cosmo)
        kwargs_special = {"D_dt": lens_cosmo.ddt}

        # make instance of LensModel class
        from lenstronomy.LensModel.lens_model import LensModel

        lensModel = LensModel(
            lens_model_list=lens_model_list,
            cosmo=cosmo,
            z_lens=z_lens,
            z_source_convention=z_source_convention,
            z_source=z_source_convention,
        )

        # make instance of LensEquationSolver to solve the lens equation
        from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

        lensEquationSolver = LensEquationSolver(lensModel=lensModel)
        (
            x_img_list,
            y_img_list,
            flux_ratio_list,
            flux_ratio_list_err,
            dt_list,
            dt_list_err,
        ) = ([], [], [], [], [], [])
        kwargs_ps = []
        for i in range(num_sources):
            z_source = z_sources[i]
            lensEquationSolver.lensModel.change_source_redshift(z_source=z_source)
            x_img_i, y_img_i = lensEquationSolver.image_position_from_source(
                kwargs_lens=kwargs_lens, sourcePos_x=x_source, sourcePos_y=y_source
            )
            mag_i = lensModel.magnification(x_img_i, y_img_i, kwargs_lens)
            lensModel.change_source_redshift(z_source=z_source)
            mag_i_ = lensModel.magnification(x_img_i, y_img_i, kwargs_lens)
            npt.assert_almost_equal(mag_i, mag_i_, decimal=8)

            dt_i = lensModel.arrival_time(x_img_i, y_img_i, kwargs_lens)
            x_img_list.append(x_img_i)
            y_img_list.append(y_img_i)
            flux_ratio_list.append(np.abs(mag_i[1:] / mag_i[0]))
            flux_ratio_list_err.append(np.ones_like(mag_i[1:]) * 0.1)
            dt_list.append(dt_i[1:] - dt_i[0])
            dt_list_err.append(np.ones_like(dt_i[1:]) * 0.1)
            kwargs_ps.append({"ra_image": x_img_i, "dec_image": y_img_i})

        time_delay_likelihood = True  # bool, set this True or False depending on whether time-delay information is available and you want to make use of its information content.
        flux_ratio_likelihood = True  # bool, modeling the flux ratios of the images
        image_position_likelihood = True  # bool, evaluating the image position likelihood (in combination with astrometric errors)

        kwargs_flux_compute = {"source_type": "INF"}
        astrometry_sigma = 0.005

        kwargs_likelihood = {
            "image_position_uncertainty": astrometry_sigma,  # astrometric uncertainty of image positions
            "image_position_likelihood": image_position_likelihood,  # evaluate point source likelihood given the measured image positions
            "time_delay_likelihood": time_delay_likelihood,  # evaluating the time-delay likelihood
            "flux_ratio_likelihood": flux_ratio_likelihood,  # enables the flux ratio likelihood
            "kwargs_flux_compute": kwargs_flux_compute,  # source_type='INF' will lead to point source
            "check_bounds": True,  # check parameter bounds and punish them
        }

        kwargs_likelihood["source_position_tolerance"] = 0.05
        kwargs_likelihood["source_position_sigma"] = 0.005

        kwargs_likelihood["source_position_likelihood"] = False
        kwargs_likelihood["image_position_uncertainty"] = astrometry_sigma

        kwargs_data = {
            "time_delays_measured": dt_list,
            "time_delays_uncertainties": dt_list_err,
            "flux_ratios": flux_ratio_list,
            "flux_ratio_errors": flux_ratio_list_err,
            "ra_image_list": x_img_list,
            "dec_image_list": y_img_list,
        }
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "z_lens": z_lens,
            "z_source_convention": z_source_convention,
            "point_source_redshift_list": z_sources,
            "point_source_model_list": point_source_model_list,
            "cosmo": cosmo,
        }

        param_class = Param(kwargs_model=kwargs_model)

        likelihood = LikelihoodModule(
            kwargs_data_joint=kwargs_data,
            kwargs_model=kwargs_model,
            param_class=param_class,
            **kwargs_likelihood,
        )
        kwargs_truth = {
            "kwargs_lens": kwargs_lens,
            "kwargs_ps": kwargs_ps,
            "kwargs_special": kwargs_special,
        }
        log_l = likelihood.log_likelihood(kwargs_return=kwargs_truth, verbose=True)
        npt.assert_almost_equal(log_l, 0, decimal=5)


if __name__ == "__main__":
    pytest.main()

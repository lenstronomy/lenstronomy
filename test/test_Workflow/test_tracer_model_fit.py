import numpy.testing as npt
import numpy as np


class TestTracerModelFit(object):
    def setup_method(self):
        # imagng data specifics
        background_rms = 0.005  # background noise per pixel
        exp_time = 500.0  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 60  # cutout pixel size per axis
        pixel_scale = 0.05  # pixel size in arcsec (area per pixel = pixel_scale**2)
        fwhm = 0.05  # full width at half maximum of PSF
        psf_type = "GAUSSIAN"  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # tracer measurements specifics
        tracer_noise_map = np.ones(
            (numPix, numPix)
        )  # variance of metallicity measurement for each pixel

        # lensing quantities
        lens_model_list = ["SIE", "SHEAR"]
        kwargs_sie = {
            "theta_E": 0.66,
            "center_x": 0.05,
            "center_y": 0,
            "e1": 0.07,
            "e2": -0.03,
        }  # parameters of the deflector lens model
        kwargs_shear = {
            "gamma1": 0.0,
            "gamma2": -0.05,
        }  # shear values to the source plane

        kwargs_lens = [kwargs_sie, kwargs_shear]
        from lenstronomy.LensModel.lens_model import LensModel

        lens_model_class = LensModel(lens_model_list)

        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {
            "amp": 16,
            "R_sersic": 0.1,
            "n_sersic": 1,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.1,
            "center_y": 0,
        }
        source_model_list = ["SERSIC_ELLIPSE"]
        kwargs_source = [kwargs_sersic]

        from lenstronomy.LightModel.light_model import LightModel

        source_model_class = LightModel(source_model_list)

        # Source tracer model
        tracer_source_model_list = ["LINEAR"]
        kwargs_tracer_source = [
            {
                "amp": 100,
                "k": 1,
                "center_x": kwargs_sersic["center_x"],
                "center_y": kwargs_sersic["center_y"],
            }
        ]
        tracer_source_class = LightModel(tracer_source_model_list)

        # lens light model
        kwargs_sersic_lens = {
            "amp": 16,
            "R_sersic": 0.6,
            "n_sersic": 2,
            "e1": -0.1,
            "e2": 0.1,
            "center_x": 0.05,
            "center_y": 0,
        }

        lens_light_model_list = ["SERSIC_ELLIPSE"]
        kwargs_lens_light = [kwargs_sersic_lens]
        lens_light_model_class = LightModel(lens_light_model_list)

        # import main simulation class of lenstronomy
        from lenstronomy.Util import util
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        import lenstronomy.Util.image_util as image_util
        from lenstronomy.ImSim.image_model import ImageModel

        # generate the coordinate grid and image properties (we only read out the relevant lines we need)
        (
            _,
            _,
            ra_at_xy_0,
            dec_at_xy_0,
            _,
            _,
            Mpix2coord,
            _,
        ) = util.make_grid_with_coordtransform(
            numPix=numPix,
            deltapix=pixel_scale,
            center_ra=0,
            center_dec=0,
            subgrid_res=1,
            inverse=False,
        )

        kwargs_image_data = {
            "background_rms": background_rms,  # rms of background noise
            "exposure_time": exp_time,  # exposure time (or a map per pixel)
            "ra_at_xy_0": ra_at_xy_0,  # RA at (0,0) pixel
            "dec_at_xy_0": dec_at_xy_0,  # DEC at (0,0) pixel
            "transform_pix2angle": Mpix2coord,
            # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
            "image_data": np.zeros((numPix, numPix))
            # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
        }

        image_data_class = ImageData(**kwargs_image_data)
        # generate the psf variables
        kwargs_psf = {
            "psf_type": "GAUSSIAN",
            "fwhm": fwhm,
            "pixel_size": pixel_scale,
            "truncation": 3,
        }

        # if you are using a PSF estimate from e.g. a star in the FoV of your exposure, you can set
        # kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': deltaPix, 'kernel_point_source': 'odd numbered 2d grid with centered star/PSF model'}

        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {
            "supersampling_factor": 1,
            "supersampling_convolution": False,
        }

        imageModel = ImageModel(
            image_data_class,
            psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            kwargs_numerics=kwargs_numerics,
        )

        # generate image
        image_model = imageModel.image(
            kwargs_lens,
            kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_ps=None,
        )

        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        image_real = image_model + poisson + bkg

        image_data_class.update_data(image_real)
        kwargs_image_data["image_data"] = image_real

        # Tracer data

        kwargs_tracer_data = {
            "noise_map": tracer_noise_map,  # variance of pixels
            "ra_at_xy_0": ra_at_xy_0,  # RA at (0,0) pixel
            "dec_at_xy_0": dec_at_xy_0,  # DEC at (0,0) pixel
            "transform_pix2angle": Mpix2coord,
            # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
            "image_data": np.zeros((numPix, numPix))
            # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
        }

        tracer_data_class = ImageData(**kwargs_tracer_data)

        from lenstronomy.ImSim.tracer_model import TracerModelSource

        tracer_model_class = TracerModelSource(
            data_class=tracer_data_class,
            tracer_source_class=tracer_source_class,
            psf_class=psf_class,
            lens_model_class=lens_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
        )

        tracer_model = tracer_model_class.tracer_model(
            kwargs_tracer_source, kwargs_lens, kwargs_source
        )

        # add noise
        tracer_noise = image_util.add_background(
            image_model, sigma_bkd=tracer_noise_map
        )
        tracer_real = tracer_model + tracer_noise

        tracer_data_class.update_data(tracer_real)
        kwargs_tracer_data["image_data"] = tracer_real

        # model fitting

        # lens models
        fixed_lens = []
        kwargs_lens_init = []
        kwargs_lens_sigma = []
        kwargs_lower_lens = []
        kwargs_upper_lens = []

        fixed_lens.append(
            {}
        )  # for this example, we fix the power-law index of the lens model to be isothermal
        kwargs_lens_init.append(
            {"theta_E": 0.7, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}
        )
        kwargs_lens_sigma.append(
            {"theta_E": 0.2, "e1": 0.05, "e2": 0.05, "center_x": 0.05, "center_y": 0.05}
        )
        kwargs_lower_lens.append(
            {"theta_E": 0.01, "e1": -0.5, "e2": -0.5, "center_x": -10, "center_y": -10}
        )
        kwargs_upper_lens.append(
            {"theta_E": 10.0, "e1": 0.5, "e2": 0.5, "center_x": 10, "center_y": 10}
        )

        fixed_lens.append({"ra_0": 0, "dec_0": 0})
        kwargs_lens_init.append({"gamma1": 0.0, "gamma2": 0.0})
        kwargs_lens_sigma.append({"gamma1": 0.1, "gamma2": 0.1})
        kwargs_lower_lens.append({"gamma1": -0.2, "gamma2": -0.2})
        kwargs_upper_lens.append({"gamma1": 0.2, "gamma2": 0.2})

        lens_params = [
            kwargs_lens_init,
            kwargs_lens_sigma,
            fixed_lens,
            kwargs_lower_lens,
            kwargs_upper_lens,
        ]

        fixed_source = []
        kwargs_source_init = []
        kwargs_source_sigma = []
        kwargs_lower_source = []
        kwargs_upper_source = []

        fixed_source.append({})
        kwargs_source_init.append(
            {
                "R_sersic": 0.2,
                "n_sersic": 1,
                "e1": 0,
                "e2": 0,
                "center_x": 0.0,
                "center_y": 0,
                "amp": 16,
            }
        )
        kwargs_source_sigma.append(
            {
                "n_sersic": 0.5,
                "R_sersic": 0.1,
                "e1": 0.05,
                "e2": 0.05,
                "center_x": 0.2,
                "center_y": 0.2,
                "amp": 10,
            }
        )
        kwargs_lower_source.append(
            {
                "e1": -0.5,
                "e2": -0.5,
                "R_sersic": 0.001,
                "n_sersic": 0.5,
                "center_x": -10,
                "center_y": -10,
                "amp": 0,
            }
        )
        kwargs_upper_source.append(
            {
                "e1": 0.5,
                "e2": 0.5,
                "R_sersic": 10,
                "n_sersic": 5.0,
                "center_x": 10,
                "center_y": 10,
                "amp": 100,
            }
        )

        source_params = [
            kwargs_source_init,
            kwargs_source_sigma,
            fixed_source,
            kwargs_lower_source,
            kwargs_upper_source,
        ]

        fixed_lens_light = []
        kwargs_lens_light_init = []
        kwargs_lens_light_sigma = []
        kwargs_lower_lens_light = []
        kwargs_upper_lens_light = []

        fixed_lens_light.append({})
        kwargs_lens_light_init.append(
            {
                "R_sersic": 0.5,
                "n_sersic": 2,
                "e1": 0,
                "e2": 0,
                "center_x": 0.0,
                "center_y": 0,
                "amp": 16,
            }
        )
        kwargs_lens_light_sigma.append(
            {
                "n_sersic": 1,
                "R_sersic": 0.3,
                "e1": 0.05,
                "e2": 0.05,
                "center_x": 0.1,
                "center_y": 0.1,
                "amp": 10,
            }
        )
        kwargs_lower_lens_light.append(
            {
                "e1": -0.5,
                "e2": -0.5,
                "R_sersic": 0.001,
                "n_sersic": 0.5,
                "center_x": -10,
                "center_y": -10,
                "amp": 0,
            }
        )
        kwargs_upper_lens_light.append(
            {
                "e1": 0.5,
                "e2": 0.5,
                "R_sersic": 10,
                "n_sersic": 5.0,
                "center_x": 10,
                "center_y": 10,
                "amp": 100,
            }
        )

        lens_light_params = [
            kwargs_lens_light_init,
            kwargs_lens_light_sigma,
            fixed_lens_light,
            kwargs_lower_lens_light,
            kwargs_upper_lens_light,
        ]

        # Tracer parameter configurations

        fixed_tracer_source = [{}]
        kwargs_tracer_source_init = [
            {"amp": 100, "k": 1, "center_x": 0.0, "center_y": 0}
        ]
        kwargs_tracer_source_sigma = [
            {"amp": 50, "k": 1, "center_x": 0.2, "center_y": 0.2}
        ]
        kwargs_lower_tracer_source = [
            {"amp": 0, "k": -2, "center_x": -5, "center_y": -5}
        ]
        kwargs_upper_tracer_source = [
            {"amp": 1000, "k": 5, "center_x": 5, "center_y": 5}
        ]

        tracer_source_params = [
            kwargs_tracer_source_init,
            kwargs_tracer_source_sigma,
            fixed_tracer_source,
            kwargs_lower_tracer_source,
            kwargs_upper_tracer_source,
        ]

        kwargs_params = {
            "lens_model": lens_params,
            "source_model": source_params,
            "lens_light_model": lens_light_params,
            "tracer_source_model": tracer_source_params,
        }

        kwargs_likelihood = {"source_marg": False}
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "lens_light_model_list": lens_light_model_list,
            "tracer_source_model_list": tracer_source_model_list,
            "tracer_source_band": 0,
        }  # what imaging band's surface brightness solution is used for tracer models

        multi_band_list = [[kwargs_image_data, kwargs_psf, kwargs_numerics]]
        # if you have multiple  bands to be modeled simultaneously, you can append them to the mutli_band_list
        kwargs_data_joint = {
            "multi_band_list": multi_band_list,
            "multi_band_type": "single-band",
            # 'multi-linear': every imaging band has independent solutions of the surface brightness, 'joint-linear': there is one joint solution of the linear coefficients demanded across the bands.
            "tracer_data": [kwargs_tracer_data, kwargs_psf, kwargs_numerics],
        }
        kwargs_constraints = {
            "linear_solver": True,
            # optional, if 'linear_solver': False, lenstronomy does not apply a linear inversion of the 'amp' parameters during fitting but instead samples them.
            "joint_source_light_with_tracer": [[0, 0, ["center_x", "center_y"]]],
        }  # link source light with tracer [[i_source_light, k_tracer_source, ['param_name1', 'param_name2', ...]], [...], ...]
        self.kwargs_data_joint = kwargs_data_joint
        self.kwargs_model = kwargs_model
        self.kwargs_constraints = kwargs_constraints
        self.kwargs_likelihood = kwargs_likelihood
        self.kwargs_params = kwargs_params

    def test_run_fit(self):
        from lenstronomy.Workflow.fitting_sequence import FittingSequence

        fitting_seq = FittingSequence(
            self.kwargs_data_joint,
            self.kwargs_model,
            self.kwargs_constraints,
            self.kwargs_likelihood,
            self.kwargs_params,
        )

        fitting_kwargs_list = [
            [
                "update_settings",
                {
                    "tracer_source_add_fixed": [
                        [0, ["amp", "k"]]
                    ],  # add tracer fixed to avoid sampling
                    "kwargs_likelihood": {
                        "tracer_likelihood": False
                    },  # remove tracer data to be fitted
                },
            ],
            ["PSO", {"sigma_scale": 1.0, "n_particles": 10, "n_iterations": 10}],
            [
                "update_settings",
                {
                    "tracer_source_remove_fixed": [
                        [0, ["amp", "k"]]
                    ],  # remove tracer fixed to sample
                    "kwargs_likelihood": {"tracer_likelihood": True},
                },
            ],  # evaluate tracer data likelihood
            ["PSO", {"sigma_scale": 0.1, "n_particles": 10, "n_iterations": 10}],
            # ['MCMC', {'n_burn': 10, 'n_run': 10, 'n_walkers': 10, 'sigma_scale': .1}]
        ]

        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        npt.assert_almost_equal(
            kwargs_result["kwargs_tracer_source"][0]["k"], 1, decimal=1
        )

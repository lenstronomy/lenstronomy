from lenstronomy.Plots.tracer_plot import TracerPlot
import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

class TestTracerPlot(object):

    def setup_method(self):
        # imagng data specifics
        background_rms = .005  # background noise per pixel
        exp_time = 500.  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 60  # cutout pixel size per axis
        pixel_scale = 0.05  # pixel size in arcsec (area per pixel = pixel_scale**2)
        fwhm = 0.05  # full width at half maximum of PSF
        psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # tracer measurements specifics
        tracer_noise_map = np.ones((numPix, numPix))  # variance of metallicity measurement for each pixel

        # lensing quantities
        lens_model_list = ['SIE', 'SHEAR']
        kwargs_sie = {'theta_E': .66, 'center_x': 0.05, 'center_y': 0, 'e1': .07,
                      'e2': -0.03}  # parameters of the deflector lens model
        kwargs_shear = {'gamma1': 0.0, 'gamma2': -0.05}  # shear values to the source plane

        kwargs_lens = [kwargs_sie, kwargs_shear]
        from lenstronomy.LensModel.lens_model import LensModel
        lens_model_class = LensModel(lens_model_list)

        # Sersic parameters in the initial simulation for the source
        kwargs_sersic = {'amp': 16, 'R_sersic': 0.1, 'n_sersic': 1, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.1,
                         'center_y': 0}
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [kwargs_sersic]

        from lenstronomy.LightModel.light_model import LightModel
        source_model_class = LightModel(source_model_list)

        # Source tracer model
        tracer_source_model_list = ['LINEAR']
        kwargs_tracer_source = [
            {'amp': 100, 'k': 1, 'center_x': kwargs_sersic['center_x'], 'center_y': kwargs_sersic['center_y']}]
        tracer_source_class = LightModel(tracer_source_model_list)

        # lens light model
        kwargs_sersic_lens = {'amp': 16, 'R_sersic': 0.6, 'n_sersic': 2, 'e1': -0.1, 'e2': 0.1, 'center_x': 0.05,
                              'center_y': 0}

        lens_light_model_list = ['SERSIC_ELLIPSE']
        kwargs_lens_light = [kwargs_sersic_lens]
        lens_light_model_class = LightModel(lens_light_model_list)

        # import main simulation class of lenstronomy
        from lenstronomy.Util import util
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        import lenstronomy.Util.image_util as image_util
        from lenstronomy.ImSim.image_model import ImageModel

        # generate the coordinate grid and image properties (we only read out the relevant lines we need)
        _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(numPix=numPix,
                                                                                                deltapix=pixel_scale,
                                                                                                center_ra=0,
                                                                                                center_dec=0,
                                                                                                subgrid_res=1,
                                                                                                inverse=False)

        kwargs_image_data = {'background_rms': background_rms,  # rms of background noise
                             'exposure_time': exp_time,  # exposure time (or a map per pixel)
                             'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                             'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel
                             'transform_pix2angle': Mpix2coord,
                             # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
                             'image_data': np.zeros((numPix, numPix))
                             # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
                             }

        image_data_class = ImageData(**kwargs_image_data)
        # generate the psf variables
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'pixel_size': pixel_scale, 'truncation': 3}

        # if you are using a PSF estimate from e.g. a star in the FoV of your exposure, you can set
        # kwargs_psf = {'psf_type': 'PIXEL', 'pixel_size': deltaPix, 'kernel_point_source': 'odd numbered 2d grid with centered star/PSF model'}

        psf_class = PSF(**kwargs_psf)
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        imageModel = ImageModel(image_data_class, psf_class, lens_model_class=lens_model_class,
                                source_model_class=source_model_class, lens_light_model_class=lens_light_model_class,
                                kwargs_numerics=kwargs_numerics)

        # generate image
        image_model = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light, kwargs_ps=None)

        poisson = image_util.add_poisson(image_model, exp_time=exp_time)
        bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
        image_real = image_model + poisson + bkg

        image_data_class.update_data(image_real)
        kwargs_image_data['image_data'] = image_real

        # Tracer data

        kwargs_tracer_data = {'noise_map': tracer_noise_map,  # variance of pixels
                              'ra_at_xy_0': ra_at_xy_0,  # RA at (0,0) pixel
                              'dec_at_xy_0': dec_at_xy_0,  # DEC at (0,0) pixel
                              'transform_pix2angle': Mpix2coord,
                              # matrix to translate shift in pixel in shift in relative RA/DEC (2x2 matrix). Make sure it's units are arcseconds or the angular units you want to model.
                              'image_data': np.zeros((numPix, numPix))
                              # 2d data vector, here initialized with zeros as place holders that get's overwritten once a simulated image with noise is created.
                              }

        tracer_data_class = ImageData(**kwargs_tracer_data)

        from lenstronomy.ImSim.tracer_model import TracerModelSource
        tracer_model_class = TracerModelSource(data_class=tracer_data_class, tracer_source_class=tracer_source_class,
                                               psf_class=psf_class, lens_model_class=lens_model_class,
                                               source_model_class=source_model_class,
                                               lens_light_model_class=lens_light_model_class)

        tracer_model = tracer_model_class.tracer_model(kwargs_tracer_source, kwargs_lens, kwargs_source)

        # add noise
        tracer_noise = image_util.add_background(image_model, sigma_bkd=tracer_noise_map)
        tracer_real = tracer_model + tracer_noise

        tracer_data_class.update_data(tracer_real)
        kwargs_tracer_data['image_data'] = tracer_real

        kwargs_likelihood = {'source_marg': False}
        kwargs_model = {'lens_model_list': lens_model_list, 'source_light_model_list': source_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'tracer_source_model_list': tracer_source_model_list,
                        'tracer_source_band': 0}  # what imaging band's surface brightness solution is used for tracer models

        multi_band_list = [[kwargs_image_data, kwargs_psf, kwargs_numerics]]
        # if you have multiple  bands to be modeled simultaneously, you can append them to the mutli_band_list
        kwargs_data_joint = {'multi_band_list': multi_band_list,
                             'multi_band_type': 'single-band',
                             # 'multi-linear': every imaging band has independent solutions of the surface brightness, 'joint-linear': there is one joint solution of the linear coefficients demanded across the bands.
                             'tracer_data': [kwargs_tracer_data, kwargs_psf, kwargs_numerics],
                             }
        self.kwargs_data_joint = kwargs_data_joint
        self.kwargs_model = kwargs_model
        self.kwargs_likelihood = kwargs_likelihood
        self.kwargs_params = {'kwargs_lens': kwargs_lens, 'kwargs_source': kwargs_source,
                         'kwargs_lens_light': kwargs_lens_light, 'kwargs_tracer_source': kwargs_tracer_source}

    def test_tracer_plot(self):

        tracer_plot = TracerPlot(self.kwargs_data_joint, self.kwargs_model, self.kwargs_params,
                                 self.kwargs_likelihood,
                                 arrow_size=0.02, cmap_string="gist_heat",
                                 fast_caustic=True)

        f, axes = plt.subplots(2, 3, figsize=(16, 8))

        tracer_plot.data_plot(ax=axes[0, 0])
        tracer_plot.model_plot(ax=axes[0, 1])
        tracer_plot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
        tracer_plot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100, plot_scale='log', v_min=None,
                                v_max=None)
        tracer_plot.convergence_plot(ax=axes[1, 1], v_max=1)
        tracer_plot.magnification_plot(ax=axes[1, 2])
        f.tight_layout()
        f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
        plt.close()
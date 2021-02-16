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
import matplotlib.pyplot as plt

from lenstronomy.Plots.multi_patch_plot import MultiPatchPlot


class TestMultiPatchPlot(object):

    def setup(self):
        # data specifics
        sigma_bkg = .05  # background noise per pixel (Gaussian)
        exp_time = 100.  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
        psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'

        # generate the coordinate grid and image properties
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        kwargs_data['exposure_time'] = exp_time * np.ones_like(kwargs_data['image_data'])
        data_class = ImageData(**kwargs_data)
        # generate the psf variables

        kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'fwhm': fwhm}
        # kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)
        psf_class = PSF(**kwargs_psf)

        # lensing quantities
        kwargs_shear = {'gamma1': 0.02, 'gamma2': -0.04}  # shear values to the source plane
        kwargs_spemd = {'theta_E': 1.26, 'gamma': 2., 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1,
                        'e2': 0.05}  # parameters of the deflector lens model

        # the lens model is a supperposition of an elliptical lens model with external shear
        lens_model_list = ['EPL', 'SHEAR']
        kwargs_lens_true = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        # choice of source type
        source_type = 'SERSIC'  # 'SERSIC' or 'SHAPELETS'

        source_x = 0.
        source_y = 0.05

        # Sersic parameters in the initial simulation
        phi_G, q = 0.5, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        kwargs_sersic_source = {'amp': 1000, 'R_sersic': 0.05, 'n_sersic': 1, 'e1': e1, 'e2': e2, 'center_x': source_x,
                                'center_y': source_y}
        # kwargs_else = {'sourcePos_x': source_x, 'sourcePos_y': source_y, 'quasar_amp': 400., 'gamma1_foreground': 0.0, 'gamma2_foreground':-0.0}
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_true = [kwargs_sersic_source]
        source_model_class = LightModel(light_model_list=source_model_list)

        lensEquationSolver = LensEquationSolver(lens_model_class)
        x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens_true, numImages=4,
                                                              min_distance=deltaPix, search_window=numPix * deltaPix)
        mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens_true)

        kwargs_numerics = {'supersampling_factor': 1}

        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                kwargs_numerics=kwargs_numerics)

        # generate image
        model = imageModel.image(kwargs_lens_true, kwargs_source_true)
        poisson = image_util.add_poisson(model, exp_time=exp_time)
        bkg = image_util.add_background(model, sigma_bkd=sigma_bkg)
        image_sim = model + bkg + poisson

        data_class.update_data(image_sim)
        kwargs_data['image_data'] = image_sim

        kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list,
                        }

        # make cutous and data instances of them
        x_pos, y_pos = data_class.map_coord2pix(x_image, y_image)
        ra_grid, dec_grid = data_class.pixel_coordinates

        multi_band_list = []
        for i in range(len(x_pos)):
            n_cut = 12
            x_c = int(x_pos[i])
            y_c = int(y_pos[i])
            image_cut = image_sim[int(y_c - n_cut):int(y_c + n_cut), int(x_c - n_cut):int(x_c + n_cut)]
            exposure_map_cut = data_class.exposure_map[int(y_c - n_cut):int(y_c + n_cut),
                               int(x_c - n_cut):int(x_c + n_cut)]
            kwargs_data_i = {
                'background_rms': data_class.background_rms,
                'exposure_time': exposure_map_cut,
                'ra_at_xy_0': ra_grid[y_c - n_cut, x_c - n_cut], 'dec_at_xy_0': dec_grid[y_c - n_cut, x_c - n_cut],
                'transform_pix2angle': data_class.transform_pix2angle
                , 'image_data': image_cut
            }
            multi_band_list.append([kwargs_data_i, kwargs_psf, kwargs_numerics])

        kwargs_params = {'kwargs_lens': kwargs_lens_true, 'kwargs_source': kwargs_source_true}
        self.multiPatch = MultiPatchPlot(multi_band_list, kwargs_model, kwargs_params, multi_band_type='joint-linear',
                 kwargs_likelihood=None, verbose=True, cmap_string="gist_heat")
        self.data_class = data_class
        self.model = model
        self.lens_model_class = lens_model_class
        self.kwargs_lens = kwargs_lens_true

    def test_data_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.data_plot(ax)
        plt.close()

    def test_model_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.model_plot(ax)
        plt.close()

    def test_source_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.source_plot(ax, delta_pix=0.01, num_pix=50, center=None)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.source_plot(ax, delta_pix=0.01, num_pix=50, center=[0, 0])
        plt.close()

    def test_normalized_residual_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.normalized_residual_plot(ax)
        plt.close()

    def test_convergence_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.convergence_plot(ax)
        plt.close()

    def test_magnification_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax = self.multiPatch.magnification_plot(ax)
        plt.close()

    def test_main_plot(self):
        f, axes = self.multiPatch.plot_main()
        plt.close()


if __name__ == '__main__':
    pytest.main()

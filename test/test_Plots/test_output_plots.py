__author__ = 'sibirrer'

import pytest
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import Data
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Plots.output_plots import LensModelPlot
import lenstronomy.Plots.output_plots as output_plots
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class TestOutputPlots(object):
    """
    test the fitting sequences
    """
    def setup(self):
        self.SimAPI = Simulation()

        # data specifics
        sigma_bkg = 0.05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        data_class = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        psf_class = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix,
                                               truncate=3,
                                               kernel=None)
        psf_class = self.SimAPI.psf_configure(psf_type='PIXEL', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix,
                                                    truncate=6,
                                                    kernel=psf_class.kernel_point_source)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        e1, e2 = param_util.phi_q2_ellipticity(0.2, 0.8)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        self.LensModel = lens_model_class
        # list of light profiles (for lens and source)
        # 'SERSIC': spherical Sersic profile
        kwargs_sersic = {'amp': 1., 'R_sersic': 0.1, 'n_sersic': 2, 'center_x': 0, 'center_y': 0}
        # 'SERSIC_ELLIPSE': elliptical Sersic profile
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {'amp': 1., 'R_sersic': .6, 'n_sersic': 7, 'center_x': 0, 'center_y': 0,
                                 'e1': e1, 'e2': e2}

        lens_light_model_list = ['SERSIC']
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{'ra_source': 0.0, 'dec_source': 0.0,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_list = ['SOURCE_POSITION']
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[True])
        kwargs_numerics = {'subgrid_res': 1, 'psf_subgrid': False}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = self.SimAPI.simulate(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)

        data_class.update_data(image_sim)
        self.kwargs_data = data_class.constructor_kwargs()
        self.kwargs_psf = psf_class.constructor_kwargs()
        self.kwargs_model = {'lens_model_list': lens_model_list,
                               'source_light_model_list': source_model_list,
                               'lens_light_model_list': lens_light_model_list,
                               'point_source_model_list': point_source_list,
                               'fixed_magnification_list': [False],
                             }
        self.kwargs_numerics = kwargs_numerics
        self.data_class = Data(self.kwargs_data)

    def test_lensModelPlot(self):

        lensPlot = LensModelPlot(self.kwargs_data, self.kwargs_psf, self.kwargs_numerics, self.kwargs_model,
                                     self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps,
                                     arrow_size=0.02, cmap_string="gist_heat", high_res=5)

        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

        lensPlot.data_plot(ax=axes[0, 0])
        lensPlot.model_plot(ax=axes[0, 1])
        lensPlot.normalized_residual_plot(ax=axes[0, 2], v_min=-6, v_max=6)
        lensPlot.source_plot(ax=axes[1, 0], convolution=False, deltaPix_source=0.01, numPix=100)
        lensPlot.convergence_plot(ax=axes[1, 1], v_max=1)
        lensPlot.magnification_plot(ax=axes[1, 2])
        plt.close()

        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

        lensPlot.decomposition_plot(ax=axes[0, 0], text='Lens light', lens_light_add=True, unconvolved=True)
        lensPlot.decomposition_plot(ax=axes[1, 0], text='Lens light convolved', lens_light_add=True)
        lensPlot.decomposition_plot(ax=axes[0, 1], text='Source light', source_add=True, unconvolved=True)
        lensPlot.decomposition_plot(ax=axes[1, 1], text='Source light convolved', source_add=True)
        lensPlot.decomposition_plot(ax=axes[0, 2], text='All components', source_add=True, lens_light_add=True,
                                        unconvolved=True)
        lensPlot.decomposition_plot(ax=axes[1, 2], text='All components convolved', source_add=True,
                                        lens_light_add=True, point_source_add=True)
        plt.close()

        f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

        lensPlot.subtract_from_data_plot(ax=axes[0,0], text='Data')
        lensPlot.subtract_from_data_plot(ax=axes[0,1], text='Data - Point Source', point_source_add=True)
        lensPlot.subtract_from_data_plot(ax=axes[0,2], text='Data - Lens Light', lens_light_add=True)
        lensPlot.subtract_from_data_plot(ax=axes[1,0], text='Data - Source Light', source_add=True)
        lensPlot.subtract_from_data_plot(ax=axes[1,1], text='Data - Source Light - Point Source', source_add=True, point_source_add=True)
        lensPlot.subtract_from_data_plot(ax=axes[1,2], text='Data - Lens Light - Point Source', lens_light_add=True, point_source_add=True)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.deflection_plot(ax=ax)
        plt.close()

        numPix = 100
        deltaPix_source = 0.01
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.error_map_source_plot(ax, numPix, deltaPix_source)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        lensPlot.absolute_residual_plot(ax=ax)
        plt.close()

    def test_lens_model_plot(self):
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        numPix = 100
        deltaPix = 0.05
        ax = output_plots.lens_model_plot(ax, self.LensModel, self.kwargs_lens, numPix, deltaPix, sourcePos_x=0, sourcePos_y=0, point_source=True)
        plt.close()

    def test_psf_iteration_compare(self):
        kwargs_psf = self.kwargs_psf
        kwargs_psf['kernel_point_source_init'] = kwargs_psf['kernel_point_source']
        f, ax = output_plots.psf_iteration_compare(kwargs_psf=kwargs_psf)
        plt.close()

    def test_external_shear_direction(self):
        f, ax = output_plots.ext_shear_direction(data_class=self.data_class, lens_model_class=self.LensModel, kwargs_lens=self.kwargs_lens,
                        strength_multiply=10)
        plt.close()

    def test_plot_chain(self):
        X2_list = [1, 1, 2]
        pos_list = [[1, 0], [2, 0], [3, 0]]
        vel_list = [[-1, 0], [0, 0], [1, 0]]
        param_list = ['test1', 'test2']
        chain = X2_list, pos_list, vel_list, None
        output_plots.plot_chain(chain=chain, param_list=param_list)
        plt.close()


if __name__ == '__main__':
    pytest.main()

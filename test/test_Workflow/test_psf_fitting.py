__author__ = 'sibirrer'

import pytest
import numpy as np
import lenstronomy.Util.util as util
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.psf_fitting import PsfFitting
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF


class TestImageModel(object):
    """
    tests the source model routines
    """

    def setup(self):

        # data specifics
        sigma_bkg = 0.01  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.3  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        sigma = util.fwhm2sigma(fwhm)
        x_grid, y_grid = util.make_grid(numPix=31, deltapix=0.05)
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian
        gaussian = Gaussian()
        kernel_point_source = gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma, sigma_y=sigma,
                                                center_x=0, center_y=0)
        kernel_point_source /= np.sum(kernel_point_source)
        kernel_point_source = util.array2image(kernel_point_source)
        self.kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}

        psf_class = PSF(**self.kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'e1': 0.01, 'e2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {'theta_E': 1., 'gamma': 1.8, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}

        lens_model_list = ['SPEP', 'SHEAR']
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
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
                           'source_amp': 10.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])

        kwargs_numerics = {'supersampling_factor': 3, 'supersampling_convolution': False, 'compute_mode': 'regular',
                           'point_source_supersampling_factor': 3}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                     lens_light_model_class,
                                     point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)
        data_class.update_data(image_sim)
        self.imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class,
                                         lens_light_model_class,
                                         point_source_class, kwargs_numerics=kwargs_numerics)

        self.psf_fitting = PsfFitting(self.imageModel)

    def test_update_psf(self):
        fwhm = 0.5
        sigma = util.fwhm2sigma(fwhm)
        x_grid, y_grid = util.make_grid(numPix=31, deltapix=0.05)
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian
        gaussian = Gaussian()
        kernel_point_source = gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma, sigma_y=sigma,
                                                center_x=0, center_y=0)
        kernel_point_source /= np.sum(kernel_point_source)
        kernel_point_source = util.array2image(kernel_point_source)
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}

        kwargs_psf_iter = {'stacking_method': 'median'}
        kwargs_psf_return, improved_bool, error_map = self.psf_fitting.update_psf(kwargs_psf, self.kwargs_lens, self.kwargs_source,
                                                                       self.kwargs_lens_light, self.kwargs_ps, **kwargs_psf_iter)
        assert improved_bool
        kernel_new = kwargs_psf_return['kernel_point_source']
        kernel_true = self.kwargs_psf['kernel_point_source']
        kernel_old = kwargs_psf['kernel_point_source']
        diff_old = np.sum((kernel_old - kernel_true) ** 2)
        diff_new = np.sum((kernel_new - kernel_true) ** 2)
        assert diff_old > diff_new

    def test_update_iterative(self):
        fwhm = 0.5
        sigma = util.fwhm2sigma(fwhm)
        x_grid, y_grid = util.make_grid(numPix=31, deltapix=0.05)
        from lenstronomy.LightModel.Profiles.gaussian import Gaussian
        gaussian = Gaussian()
        kernel_point_source = gaussian.function(x_grid, y_grid, amp=1., sigma_x=sigma, sigma_y=sigma,
                                              center_x=0, center_y=0)
        kernel_point_source /= np.sum(kernel_point_source)
        kernel_point_source = util.array2image(kernel_point_source)
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel_point_source}
        kwargs_psf_iter = {'stacking_method': 'median'}
        kwargs_psf_new = self.psf_fitting.update_iterative(kwargs_psf, self.kwargs_lens, self.kwargs_source,
                                                                       self.kwargs_lens_light, self.kwargs_ps,
                                                           **kwargs_psf_iter)
        kernel_new = kwargs_psf_new['kernel_point_source']
        kernel_true = self.kwargs_psf['kernel_point_source']
        kernel_old = kwargs_psf['kernel_point_source']
        diff_old = np.sum((kernel_old - kernel_true) ** 2)
        diff_new = np.sum((kernel_new - kernel_true) ** 2)
        assert diff_old > diff_new
        assert diff_new < 0.01

        kwargs_psf_new = self.psf_fitting.update_iterative(kwargs_psf, self.kwargs_lens, self.kwargs_source,
                                                           self.kwargs_lens_light, self.kwargs_ps, num_iter=3,
                                                           no_break=True)
        kernel_new = kwargs_psf_new['kernel_point_source']
        kernel_true = self.kwargs_psf['kernel_point_source']
        kernel_old = kwargs_psf['kernel_point_source']
        diff_old = np.sum((kernel_old - kernel_true) ** 2)
        diff_new = np.sum((kernel_new - kernel_true) ** 2)
        assert diff_old > diff_new
        assert diff_new < 0.01

    def test_mask_point_source(self):
        ra_image, dec_image, amp = self.imageModel.PointSource.point_source_list(self.kwargs_ps, self.kwargs_lens)
        print(ra_image, dec_image, amp)
        x_grid, y_grid = self.imageModel.Data.pixel_coordinates
        x_grid = util.image2array(x_grid)
        y_grid = util.image2array(y_grid)
        radius = 0.5
        mask_point_source = self.psf_fitting.mask_point_source(ra_image, dec_image, x_grid, y_grid, radius, i=0)
        assert mask_point_source[10, 10] == 1


if __name__ == '__main__':
    pytest.main()

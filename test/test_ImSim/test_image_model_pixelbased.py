__author__ = 'sibirrer'

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


class TestImageModel(object):
    """
    tests the source model routines
    """
    def setup(self):

        # data specifics
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg, inverse=True)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        kernel = psf_class.kernel_point_source
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': kernel, 'psf_error_map': np.ones_like(kernel) * 0.001}
        psf_class = PSF(**kwargs_psf)

        # 'EXERNAL_SHEAR': external shear
        kwargs_shear = {'gamma1': 0.01, 'gamma2': 0.01}  # gamma_ext: shear strength, psi_ext: shear angel (in radian)
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
        kwargs_lens_light_base = [kwargs_sersic]
        lens_light_model_class_base = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source_base = [kwargs_sersic_ellipse]
        source_model_class_base = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{'ra_source': 0.01, 'dec_source': 0.0,
                       'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class_base = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics_base = {'supersampling_factor': 2, 'supersampling_convolution': False}
        imageModel_base = ImageModel(data_class, psf_class, lens_model_class, source_model_class_base, lens_light_model_class_base, point_source_class_base, kwargs_numerics=kwargs_numerics_base)
        image_sim = sim_util.simulate_simple(imageModel_base, self.kwargs_lens, kwargs_source_base,
                                       kwargs_lens_light_base, self.kwargs_ps)
        data_class.update_data(image_sim)

        # create a starlet light distributions
        from lenstronomy.LightModel.Profiles.starlets import SLIT_Starlets
        n_scales = 6
        source_map = imageModel_base.source_surface_brightness(kwargs_source_base, de_lensed=True, unconvolved=True)
        source_map_starlets = SLIT_Starlets().decomposition_2d(source_map, n_scales)
        self.kwargs_source = [{'amp': source_map_starlets, 'n_scales': n_scales, 'n_pixels': numPix, 'scale': deltaPix, 'center_x': 0, 'center_y': 0}]
        source_model_class = LightModel(light_model_list=['SLIT_STARLETS'])
        lens_light_map = imageModel_base.lens_surface_brightness(kwargs_lens_light_base, unconvolved=True)
        lens_light_starlets = SLIT_Starlets().decomposition_2d(lens_light_map, n_scales)
        self.kwargs_lens_light = [{'amp': lens_light_starlets, 'n_scales': n_scales, 'n_pixels': numPix, 'scale': deltaPix, 'center_x': 0, 'center_y': 0}]
        lens_light_model_class = LightModel(light_model_list=['SLIT_STARLETS'])

        kwargs_numerics = {'supersampling_factor': 1}
        kwargs_pixelbased = {
            'supersampling_factor_source': 2, # supersampling of pixelated source grid

            # following choices are to minimize pixel solver runtime (not to get accurate reconstruction!)
            'threshold_decrease_type': 'none',
            'num_iter_source': 2,
            'num_iter_lens': 2,
            'num_iter_global': 2,
            'num_iter_weights': 2,
        }
        self.imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, 
                                         source_model_class=source_model_class, 
                                         lens_light_model_class=lens_light_model_class, 
                                         point_source_class=None,
                                         kwargs_numerics=kwargs_numerics, kwargs_pixelbased=kwargs_pixelbased)
        self.imageModel_source = ImageLinearFit(data_class, psf_class, lens_model_class, 
                                                source_model_class=source_model_class, 
                                                lens_light_model_class=None, 
                                                point_source_class=None,
                                                kwargs_numerics=kwargs_numerics, kwargs_pixelbased=kwargs_pixelbased)
        self.imageModel_source_ps = ImageLinearFit(data_class, psf_class, lens_model_class, 
                                                   source_model_class=source_model_class, 
                                                   lens_light_model_class=None, 
                                                   point_source_class=point_source_class_base,
                                                   kwargs_numerics=kwargs_numerics, kwargs_pixelbased=kwargs_pixelbased)
        
        self.solver = LensEquationSolver(lensModel=self.imageModel.LensModel)

    def test_source_surface_brightness(self):
        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens,
                                                                 unconvolved=False, de_lensed=True)
        assert len(source_model) == 100

        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13939841209844345 * 0.05**2, decimal=4)

        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13536114618182182 * 0.05**2, decimal=4)

    def test_lens_surface_brightness(self):
        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=False)
        npt.assert_almost_equal(lens_flux[50, 50], 0.011827638372778893, decimal=4)

        # the following should raise an error: no deconvolution is performed when pixel-based modelling lens light
        # lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=True)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(self.kwargs_lens, self.kwargs_source, 
                                                                                self.kwargs_lens_light, self.kwargs_ps, 
                                                                                inv_bool=False)
        chi2_reduced = self.imageModel.reduced_chi2(model, error_map)
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_image_with_params(self):
        model = self.imageModel.image(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True)
        error_map = self.imageModel._error_map_psf(self.kwargs_lens, self.kwargs_ps)
        chi2_reduced = self.imageModel.reduced_chi2(model, error_map)
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, source_marg=False)
        npt.assert_almost_equal(logL, -5000, decimal=-3)

    def test_reduced_residuals(self):
        model = sim_util.simulate_simple(self.imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps, no_noise=True)
        residuals = self.imageModel.reduced_residuals(model, error_map=0)
        npt.assert_almost_equal(np.std(residuals), 1.01, decimal=1)

        chi2 = self.imageModel.reduced_chi2(model, error_map=0)
        npt.assert_almost_equal(chi2, 1, decimal=1)

    def test_numData_evaluate(self):
        numData = self.imageModel.num_data_evaluate
        assert numData == 10000

    def test_num_param_linear(self):
        num_param_linear = self.imageModel.num_param_linear(self.kwargs_lens, self.kwargs_source,
                                                            self.kwargs_lens_light, self.kwargs_ps)
        assert num_param_linear == 0 # pixels of pixel-based profiles not counted as linear param

        num_param_linear = self.imageModel_source.num_param_linear(self.kwargs_lens, self.kwargs_source,
                                                                   self.kwargs_lens_light, self.kwargs_ps)
        assert num_param_linear == 0 # pixels of pixel-based profiles not counted as linear param

        num_param_linear = self.imageModel_source_ps.num_param_linear(self.kwargs_lens, self.kwargs_source,
                                                                      self.kwargs_lens_light, self.kwargs_ps)
        assert num_param_linear == 1 # the point source

    def test_update_data(self):
        kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=1, exposure_time=1, background_rms=1, inverse=True)
        data_class = ImageData(**kwargs_data)
        self.imageModel.update_data(data_class)
        assert self.imageModel.Data.num_pixel == 100

    def test_create_empty(self):
        kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=1, exposure_time=1, background_rms=1)
        data_class = ImageData(**kwargs_data)
        imageModel_empty = ImageModel(data_class, PSF())
        assert imageModel_empty._psf_error_map == False

        flux = imageModel_empty.lens_surface_brightness(kwargs_lens_light=None)
        assert flux.all() == 0

    def test_extinction_map(self):
        kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=1, exposure_time=1, background_rms=1)
        data_class = ImageData(**kwargs_data)
        extinction_class = DifferentialExtinction(optical_depth_model=['UNIFORM'], tau0_index=0)
        imageModel = ImageModel(data_class, PSF(), extinction_class=extinction_class)
        extinction = imageModel.extinction_map(kwargs_extinction=[{'amp': 1}], kwargs_special={'tau0_list': [1, 0, 0]})
        npt.assert_almost_equal(extinction, np.exp(-1))

    def test_error_response(self):
        C_D_response, psf_model_error = self.imageModel._error_response(self.kwargs_lens, self.kwargs_ps, kwargs_special=None)
        assert len(psf_model_error) == 100
        print(np.sum(psf_model_error))
        npt.assert_almost_equal(np.sum(psf_model_error), 0, decimal=3)

        C_D_response, psf_model_error = self.imageModel_source._error_response(self.kwargs_lens, self.kwargs_ps, kwargs_special=None)
        assert len(psf_model_error) == 100
        print(np.sum(psf_model_error))
        npt.assert_almost_equal(np.sum(psf_model_error), 0, decimal=3)

        C_D_response, psf_model_error = self.imageModel_source_ps._error_response(self.kwargs_lens, self.kwargs_ps, kwargs_special=None)
        assert len(psf_model_error) == 100
        print(np.sum(psf_model_error))
        npt.assert_almost_equal(np.sum(psf_model_error), 0.0019271126921470687, decimal=3)

    def test_point_source_linear_response_set(self):
        kwargs_special = {'delta_x_image': [0.1, 0.1], 'delta_y_image': [-0.1, -0.1]}
        ra_pos, dec_pos, amp, num_point = self.imageModel_source_ps.point_source_linear_response_set(self.kwargs_ps, self.kwargs_lens, kwargs_special, with_amp=True)
        ra, dec = self.imageModel_source_ps.PointSource.image_position(self.kwargs_ps, self.kwargs_lens)
        npt.assert_almost_equal(ra[0][0], ra_pos[0][0] - 0.1, decimal=5)

    def test_displace_astrometry(self):
        kwargs_special = {'delta_x_image': np.array([0.1, 0.1]), 'delta_y_image': np.array([-0.1, -0.1])}
        x_pos, y_pos = np.array([0, 0]), np.array([0, 0])
        x_shift, y_shift = self.imageModel_source_ps._displace_astrometry(x_pos, y_pos, kwargs_special=kwargs_special)
        assert x_pos[0] == 0
        assert x_shift[0] == kwargs_special['delta_x_image'][0]
        assert y_pos[0] == 0
        assert y_shift[0] == kwargs_special['delta_y_image'][0]


if __name__ == '__main__':
    pytest.main()

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
        psf_class._psf_error_map = np.zeros_like(psf_class.kernel_point_source)

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
        self.kwargs_ps = [{'ra_source': 0.01, 'dec_source': 0.0,
                       'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'supersampling_factor': 2, 'supersampling_convolution': False, 'compute_mode': 'gaussian'}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, self.kwargs_source,
                                       self.kwargs_lens_light, self.kwargs_ps)
        data_class.update_data(image_sim)

        self.imageModel = ImageLinearFit(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        self.solver = LensEquationSolver(lensModel=self.imageModel.LensModel)

    def test_source_surface_brightness(self):
        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13939841209844345 * 0.05 ** 2, decimal=4)

        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13536114618182182 * 0.05**2, decimal=4)

    def test_lens_surface_brightness(self):
        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=False)
        print(np.sum(lens_flux), 'test lens flux')
        npt.assert_almost_equal(lens_flux[50, 50], 0.0010788981265391802, decimal=4)
        #npt.assert_almost_equal(lens_flux[50, 50], 0.54214440654021534 * 0.05 ** 2, decimal=4)

        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=True)
        npt.assert_almost_equal(lens_flux[50, 50], 4.7310552067454452 * 0.05**2, decimal=4)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, inv_bool=False)
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

        logLmarg = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                                               self.kwargs_ps, source_marg=True)
        npt.assert_almost_equal(logL - logLmarg, 0, decimal=-3)

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

    def test_point_source_rendering(self):
        # initialize data

        numPix = 100
        deltaPix = 0.05
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exposure_time=1, sigma_bkg=1)
        data_class = ImageData(**kwargs_data)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {'kernel_point_source': kernel, 'psf_type': 'PIXEL'}
        psf_class = PSF(**kwargs_psf)
        lens_model_class = LensModel(['SPEP'])
        source_model_class = LightModel([])
        lens_light_model_class = LightModel([])
        kwargs_numerics =  {'supersampling_factor': 2, 'supersampling_convolution': True, 'point_source_supersampling_factor': 1}
        point_source_class = PointSource(point_source_type_list=['LENSED_POSITION'], fixed_magnification_list=[False])
        makeImage = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        # chose point source positions
        x_pix = np.array([10, 5, 10, 90])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.8)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        kwargs_else = [{'ra_image': ra_pos, 'dec_image': dec_pos, 'point_amp': np.ones_like(ra_pos)}]
        image = makeImage.image(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_ps=kwargs_else)
        #print(np.shape(model), 'test')
        #image = makeImage.ImageNumerics.array2image(model)
        for i in range(len(x_pix)):
            npt.assert_almost_equal(image[y_pix[i], x_pix[i]], 1, decimal=2)

        x_pix = np.array([10.5, 5.5, 10.5, 90.5])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        phi, q = 0., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        kwargs_else = [{'ra_image': ra_pos, 'dec_image': dec_pos, 'point_amp': np.ones_like(ra_pos)}]
        image = makeImage.image(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_ps=kwargs_else)
        #image = makeImage.ImageNumerics.array2image(model)
        for i in range(len(x_pix)):
            print(int(y_pix[i]), int(x_pix[i]+0.5))
            npt.assert_almost_equal(image[int(y_pix[i]), int(x_pix[i])], 0.5, decimal=1)
            npt.assert_almost_equal(image[int(y_pix[i]), int(x_pix[i]+0.5)], 0.5, decimal=1)

    def test_point_source(self):

        pointSource = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_ps = [{'source_amp': 1000, 'ra_source': 0.1, 'dec_source': 0.1}]
        lensModel = LensModel(lens_model_list=['SIS'])
        kwargs_lens = [{'theta_E': 1, 'center_x': 0, 'center_y': 0}]
        numPix = 64
        deltaPix = 0.13
        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exposure_time=1, sigma_bkg=1)
        data_class = ImageData(**kwargs_data)

        psf_type = "GAUSSIAN"
        fwhm = 0.9
        kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm}
        psf_class = PSF(**kwargs_psf)
        imageModel = ImageModel(data_class=data_class, psf_class=psf_class, lens_model_class=lensModel,
                                point_source_class=pointSource)
        image = imageModel.image(kwargs_lens=kwargs_lens, kwargs_ps=kwargs_ps)
        assert np.sum(image) > 0

    def test_error_map_source(self):
        sourceModel = LightModel(light_model_list=['UNIFORM', 'UNIFORM'])

        kwargs_data = sim_util.data_configure_simple(numPix=10, deltaPix=1, exposure_time=1, sigma_bkg=1)
        data_class = ImageData(**kwargs_data)

        psf_type = "GAUSSIAN"
        fwhm = 0.9
        kwargs_psf = {'psf_type': psf_type, 'fwhm': fwhm}
        psf_class = PSF(**kwargs_psf)
        imageModel = ImageLinearFit(data_class=data_class, psf_class=psf_class, lens_model_class=None,
                                point_source_class=None, source_model_class=sourceModel)

        x_grid, y_grid = util.make_grid(numPix=10, deltapix=1)
        error_map = imageModel.error_map_source(kwargs_source=[{'amp': 1}, {'amp': 1}], x_grid=x_grid, y_grid=y_grid, cov_param=np.array([[1, 0], [0, 1]]))
        assert error_map[0] == 2


if __name__ == '__main__':
    pytest.main()

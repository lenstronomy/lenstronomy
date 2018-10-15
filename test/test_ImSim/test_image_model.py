__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest

import lenstronomy.Util.param_util as param_util
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import Data
from lenstronomy.Data.psf import PSF


class TestImageModel(object):
    """
    tests the source model routines
    """
    def setup(self):
        self.SimAPI = Simulation()

        # data specifics
        sigma_bkg = .05  # background noise per pixel
        exp_time = 100  # exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
        numPix = 100  # cutout pixel size
        deltaPix = 0.05  # pixel size in arcsec (area per pixel = deltaPix**2)
        fwhm = 0.5  # full width half max of PSF

        # PSF specification

        kwargs_data = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = Data(kwargs_data)
        kwargs_psf = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix, truncate=3,
                                          kernel=None)
        psf_class = PSF(kwargs_psf)
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
        kwargs_numerics = {'subgrid_res': 2, 'psf_subgrid': True, 'psf_error_map': True}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = self.SimAPI.simulate(imageModel, self.kwargs_lens, self.kwargs_source,
                                       self.kwargs_lens_light, self.kwargs_ps)
        data_class.update_data(image_sim)

        self.imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        self.solver = LensEquationSolver(lensModel=self.imageModel.LensModel)

    def test_source_surface_brightness(self):
        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13939841209844345, decimal=4)

        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False)
        assert len(source_model) == 100
        npt.assert_almost_equal(source_model[10, 10], 0.13536114618182182, decimal=4)

    def test_lens_surface_brightness(self):
        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=False)
        npt.assert_almost_equal(lens_flux[50, 50], 0.54214440654021534, decimal=4)

        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=True)
        npt.assert_almost_equal(lens_flux[50, 50], 4.7310552067454452, decimal=4)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, inv_bool=False)
        chi2_reduced = self.imageModel.reduced_chi2(model, error_map)
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_image_with_params(self):
        model = self.imageModel.image(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True)
        error_map = self.imageModel.error_map(self.kwargs_lens, self.kwargs_ps)
        chi2_reduced = self.imageModel.reduced_chi2(model, error_map)
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_point_sources_list(self):
        point_source_list = self.imageModel.point_sources_list(self.kwargs_ps, self.kwargs_lens)
        assert len(point_source_list) == 4

    def test_image_positions(self):
        x_im, y_im = self.imageModel.image_positions(self.kwargs_ps, self.kwargs_lens)
        ra_pos, dec_pos = self.solver.image_position_from_source(sourcePos_x=self.kwargs_ps[0]['ra_source'],
                                                                 sourcePos_y=self.kwargs_ps[0]['dec_source'],
                                                                 kwargs_lens=self.kwargs_lens)
        ra_pos_new = x_im[0]
        print(ra_pos_new, ra_pos)
        npt.assert_almost_equal(ra_pos_new[0], ra_pos[0], decimal=8)
        npt.assert_almost_equal(ra_pos_new[1], ra_pos[1], decimal=8)
        npt.assert_almost_equal(ra_pos_new[2], ra_pos[2], decimal=8)
        npt.assert_almost_equal(ra_pos_new[3], ra_pos[3], decimal=8)

    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, source_marg=False)
        npt.assert_almost_equal(logL, -5000, decimal=-3)

        logLmarg = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                                               self.kwargs_ps, source_marg=True)
        npt.assert_almost_equal(logL - logLmarg, 0, decimal=-3)

    def test_reduced_residuals(self):
        model = self.SimAPI.simulate(self.imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps, no_noise=True)
        residuals = self.imageModel.reduced_residuals(model, error_map=0)
        npt.assert_almost_equal(np.std(residuals), 1.01, decimal=1)

        chi2 = self.imageModel.reduced_chi2(model, error_map=0)
        npt.assert_almost_equal(chi2, 1, decimal=1)

    def test_numData_evaluate(self):
        numData = self.imageModel.numData_evaluate()
        assert numData == 10000

    def test_fermat_potential(self):
        phi_fermat = self.imageModel.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        print(phi_fermat)
        npt.assert_almost_equal(phi_fermat[0][0], -0.2630531731871062, decimal=3)
        npt.assert_almost_equal(phi_fermat[0][1], -0.2809100018126987, decimal=3)
        npt.assert_almost_equal(phi_fermat[0][2], -0.5086643370512096, decimal=3)
        npt.assert_almost_equal(phi_fermat[0][3], -0.5131716608238992, decimal=3)

    def test_add_mask(self):
        mask = np.array([[0, 1],[1, 0]])
        A = np.ones((10, 4))
        A_masked = self.imageModel._add_mask(A, mask)
        assert A[0, 1] == A_masked[0, 1]
        assert A_masked[0, 3] == 0

    def test_point_source_rendering(self):
        # initialize data
        from lenstronomy.SimulationAPI.simulations import Simulation
        SimAPI = Simulation()
        numPix = 100
        deltaPix = 0.05
        kwargs_data = SimAPI.data_configure(numPix, deltaPix, exposure_time=1, sigma_bkg=1)
        data_class = Data(kwargs_data)
        kernel = np.zeros((5, 5))
        kernel[2, 2] = 1
        kwargs_psf = {'kernel_point_source': kernel, 'kernel_pixel': kernel, 'psf_type': 'PIXEL'}
        psf_class = PSF(kwargs_psf)
        lens_model_class = LensModel(['SPEP'])
        source_model_class = LightModel([])
        lens_light_model_class = LightModel([])
        kwargs_numerics = {'subgrid_res': 2, 'point_source_subgrid': 1}
        point_source_class = PointSource(point_source_type_list=['LENSED_POSITION'], fixed_magnification_list=[False])
        makeImage = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)
        # chose point source positions
        x_pix = np.array([10, 5, 10, 90])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.8)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        kwargs_else = [{'ra_image': ra_pos, 'dec_image': dec_pos, 'point_amp': np.ones_like(ra_pos)}]
        model = makeImage.image(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_ps=kwargs_else)
        image = makeImage.ImageNumerics.array2image(model)
        for i in range(len(x_pix)):
            npt.assert_almost_equal(image[y_pix[i], x_pix[i]], 1, decimal=2)

        x_pix = np.array([10.5, 5.5, 10.5, 90.5])
        y_pix = np.array([40, 50, 60, 50])
        ra_pos, dec_pos = makeImage.Data.map_pix2coord(x_pix, y_pix)
        phi, q = 0., 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens_init = [{'theta_E': 1, 'gamma': 2, 'e1': e1, 'e2': e2, 'center_x': 0, 'center_y': 0}]
        kwargs_else = [{'ra_image': ra_pos, 'dec_image': dec_pos, 'point_amp': np.ones_like(ra_pos)}]
        model = makeImage.image(kwargs_lens_init, kwargs_source={}, kwargs_lens_light={}, kwargs_ps=kwargs_else)
        image = makeImage.ImageNumerics.array2image(model)
        for i in range(len(x_pix)):
            print(int(y_pix[i]), int(x_pix[i]+0.5))
            npt.assert_almost_equal(image[int(y_pix[i]), int(x_pix[i])], 0.5, decimal=1)
            npt.assert_almost_equal(image[int(y_pix[i]), int(x_pix[i]+0.5)], 0.5, decimal=1)


if __name__ == '__main__':
    pytest.main()

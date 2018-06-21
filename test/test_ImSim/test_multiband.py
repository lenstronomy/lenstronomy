__author__ = 'sibirrer'

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.ImSim.multiband import Multiband
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.simulations import Simulation
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


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

        data_class = self.SimAPI.data_configure(numPix, deltaPix, exp_time, sigma_bkg)
        psf_class = self.SimAPI.psf_configure(psf_type='GAUSSIAN', fwhm=fwhm, kernelsize=31, deltaPix=deltaPix,
                                               truncate=5)

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
        self.kwargs_ps = [{'ra_source': 0.0001, 'dec_source': 0.0,
                           'source_amp': 1.}]  # quasar point source position in the source plane and intrinsic brightness
        point_source_class = PointSource(point_source_type_list=['SOURCE_POSITION'], fixed_magnification_list=[True])
        kwargs_numerics = {'subgrid_res': 2, 'psf_subgrid': True}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = self.SimAPI.simulate(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)
        data_class.update_data(image_sim)
        kwargs_data = data_class.constructor_kwargs()
        kwargs_psf = psf_class.constructor_kwargs()
        self.solver = LensEquationSolver(lensModel=lens_model_class)
        multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
        self.imageModel = Multiband(multi_band_list, lens_model_class, source_model_class, lens_light_model_class, point_source_class)

    def test_source_surface_brightness(self):
        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=False, de_lensed=False)
        assert len(source_model[0]) == 100
        npt.assert_almost_equal(source_model[0][10, 10], 0.1370014246240874, decimal=8)

        source_model = self.imageModel.source_surface_brightness(self.kwargs_source, self.kwargs_lens, unconvolved=True, de_lensed=False)
        assert len(source_model[0]) == 100
        npt.assert_almost_equal(source_model[0][10, 10], 0.13164547458291054, decimal=8)

    def test_lens_surface_brightness(self):
        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=False)
        npt.assert_almost_equal(lens_flux[0][50, 50], 0.4415194068014886, decimal=8)

        lens_flux = self.imageModel.lens_surface_brightness(self.kwargs_lens_light, unconvolved=True)
        npt.assert_almost_equal(lens_flux[0][50, 50], 4.7310552067454452, decimal=8)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, inv_bool=False)
        chi2_reduced = self.imageModel._imageModel_list[0].reduced_chi2(model[0], error_map[0])
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_image(self):
        model = self.imageModel.image(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, unconvolved=False, source_add=True, lens_light_add=True, point_source_add=True)
        error_map = self.imageModel.error_map(self.kwargs_lens, self.kwargs_ps)
        chi2_reduced = self.imageModel._imageModel_list[0].reduced_chi2(model[0], error_map[0])
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)

    def test_image_positions(self):
        x_im, y_im = self.imageModel.image_positions(self.kwargs_ps, self.kwargs_lens)
        ra_pos, dec_pos = self.solver.image_position_from_source(sourcePos_x=self.kwargs_ps[0]['ra_source'],
                                                                 sourcePos_y=self.kwargs_ps[0]['dec_source'],
                                                                 kwargs_lens=self.kwargs_lens)
        ra_pos_new = x_im[0]
        npt.assert_almost_equal(ra_pos_new[0], ra_pos[0], decimal=8)
        npt.assert_almost_equal(ra_pos_new[1], ra_pos[1], decimal=8)
        npt.assert_almost_equal(ra_pos_new[2], ra_pos[2], decimal=8)
        npt.assert_almost_equal(ra_pos_new[3], ra_pos[3], decimal=8)


    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, source_marg=False)
        npt.assert_almost_equal(logL, -5100, decimal=-3)

        logLmarg = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                                               self.kwargs_ps, source_marg=True)
        npt.assert_almost_equal(logL - logLmarg, 0, decimal=-2)

    def test_numData_evaluate(self):
        numData = self.imageModel.numData_evaluate()
        assert numData == 10000

    def test_fermat_potential(self):
        phi_fermat = self.imageModel.fermat_potential(self.kwargs_lens, self.kwargs_ps)
        phi_fermat = phi_fermat[0]
        npt.assert_almost_equal(phi_fermat[0], -0.2719737, decimal=3)
        npt.assert_almost_equal(phi_fermat[1], -0.2719737, decimal=3)
        npt.assert_almost_equal(phi_fermat[2], -0.51082354, decimal=3)
        npt.assert_almost_equal(phi_fermat[3], -0.51082354, decimal=3)


if __name__ == '__main__':
    pytest.main()



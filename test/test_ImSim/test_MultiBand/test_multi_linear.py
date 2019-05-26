__author__ = 'sibirrer'

import numpy.testing as npt
import pytest
import numpy as np

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.MultiBand.multi_linear import MultiLinear
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


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

        kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': fwhm, 'truncation': 5, 'pixel_size': deltaPix}
        psf_class = PSF(**kwargs_psf)
        # 'EXTERNAL_SHEAR': external shear
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
        kwargs_numerics = {'supersampling_factor': 2, 'supersampling_convolution': True, 'compute_mode': 'gaussian'}
        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = sim_util.simulate_simple(imageModel, self.kwargs_lens, self.kwargs_source,
                                         self.kwargs_lens_light, self.kwargs_ps)
        data_class.update_data(image_sim)
        kwargs_data['image_data'] = image_sim
        self.solver = LensEquationSolver(lensModel=lens_model_class)
        multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]
        kwargs_model = {'lens_model_list': lens_model_list, 'source_light_model_list': source_model_list,
                        'point_source_model_list': ['SOURCE_POSITION'], 'fixed_magnification_list': [True]}
        self.imageModel = MultiLinear(multi_band_list, kwargs_model, likelihood_mask_list=None, compute_bool=None)

    def test_image_linear_solve(self):
        model, error_map, cov_param, param = self.imageModel.image_linear_solve(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, inv_bool=False)
        chi2_reduced = self.imageModel._imageModel_list[0].reduced_chi2(model[0], error_map[0])
        npt.assert_almost_equal(chi2_reduced, 1, decimal=1)
        chi2_reduced_list = self.imageModel.reduced_residuals(model_list=model, error_map_list=error_map)
        npt.assert_almost_equal(np.sum(chi2_reduced_list[0]**2)/(100**2), 1, decimal=1)

    def test_likelihood_data_given_model(self):
        logL = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light, self.kwargs_ps, source_marg=False)
        npt.assert_almost_equal(logL, -5100, decimal=-3)

        logLmarg = self.imageModel.likelihood_data_given_model(self.kwargs_lens, self.kwargs_source, self.kwargs_lens_light,
                                                               self.kwargs_ps, source_marg=True)
        npt.assert_almost_equal(logL - logLmarg, 0, decimal=-2)

    def test_numData_evaluate(self):
        numData = self.imageModel.num_data_evaluate
        assert numData == 10000


if __name__ == '__main__':
    pytest.main()

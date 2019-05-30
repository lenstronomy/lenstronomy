import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Sampling.Likelihoods.flux_ratio_likelihood import FluxRatioLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestFluxRatioLikelihood(object):

    def setup(self):
        lens_model_list = ['SPEP', 'SHEAR']
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        x_source, y_source = 0.02, 0.01
        kwargs_lens = [{'theta_E': 1., 'e1': 0.1, 'e2': 0.1, 'gamma': 2., 'center_x': 0, 'center_y': 0},
                       {'e1': 0.06, 'e2': -0.03}]

        x_img, y_img = lensEquationSolver.image_position_from_source(kwargs_lens=kwargs_lens, sourcePos_x=x_source,
                                                                     sourcePos_y=y_source)
        print('image positions are: ', x_img, y_img)
        mag_inf = lensModel.magnification(x_img, y_img, kwargs_lens)
        print('point source magnification: ', mag_inf)

        source_size_arcsec = 0.001
        window_size = 0.1
        grid_number = 100
        print('source size in arcsec: ', source_size_arcsec)
        mag_finite = lensModelExtensions.magnification_finite(x_pos=x_img, y_pos=y_img, kwargs_lens=kwargs_lens,
                                                              source_sigma=source_size_arcsec, window_size=window_size,
                                                              grid_number=grid_number)
        flux_ratios = mag_finite[1:] / mag_finite[0]
        flux_ratio_errors = [0.1, 0.1, 0.1]
        self.flux_likelihood = FluxRatioLikelihood(lens_model_class=lensModel, flux_ratios=flux_ratios, flux_ratio_errors=flux_ratio_errors,
                 source_type='GAUSSIAN', window_size=window_size, grid_number=grid_number)

        self.flux_likelihood_inf = FluxRatioLikelihood(lens_model_class=lensModel, flux_ratios=flux_ratios,
                                                   flux_ratio_errors=flux_ratio_errors,
                                                   source_type='INF', window_size=window_size,
                                                   grid_number=grid_number)
        self.kwargs_cosmo = {'source_size': source_size_arcsec}
        self.x_img, self.y_img = x_img, y_img
        self.kwargs_lens = kwargs_lens

    def test_logL(self):
        logL = self.flux_likelihood.logL(self.x_img, self.y_img, self.kwargs_lens, kwargs_cosmo=self.kwargs_cosmo)
        assert logL == 0

        logL_inf = self.flux_likelihood_inf.logL(self.x_img, self.y_img, self.kwargs_lens, {})
        npt.assert_almost_equal(logL_inf, 0 , decimal=4)

    def test__logL(self):
        lensModel = LensModel(lens_model_list=[])
        flux_ratios_init = np.array([1., 1., 1.])
        flux_ratio_errors = np.array([1., 1., 1.])
        flux_likelihood = FluxRatioLikelihood(lens_model_class=lensModel, flux_ratios=flux_ratios_init,
                            flux_ratio_errors=flux_ratio_errors)

        flux_ratios = np.array([0, 1, np.nan])
        logL = flux_likelihood._logL(flux_ratios)
        assert logL == -10 ** 15

        flux_likelihood = FluxRatioLikelihood(lens_model_class=lensModel, flux_ratios=flux_ratios_init,
                                              flux_ratio_errors=np.array([0., 1., 1.]))
        flux_ratios = np.array([1., 1., 1.])
        logL = flux_likelihood._logL(flux_ratios)
        assert logL == -10 ** 15


if __name__ == '__main__':
    pytest.main()

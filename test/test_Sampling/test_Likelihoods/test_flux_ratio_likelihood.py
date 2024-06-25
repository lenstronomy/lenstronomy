import pytest
import numpy as np
import numpy.testing as npt
from lenstronomy.Sampling.Likelihoods.flux_ratio_likelihood import FluxRatioLikelihood
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestFluxRatioLikelihood(object):
    def setup_method(self):
        lens_model_list = ["SPEP", "SHEAR"]
        lensModel = LensModel(lens_model_list=lens_model_list)
        lensModelExtensions = LensModelExtensions(lensModel=lensModel)
        lensEquationSolver = LensEquationSolver(lensModel=lensModel)

        x_source, y_source = 0.02, 0.01
        x_source2, y_source2 = -0.02, 0.01
        kwargs_lens = [
            {
                "theta_E": 1.0,
                "e1": 0.1,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0,
                "center_y": 0,
            },
            {"gamma1": 0.06, "gamma2": -0.03},
        ]

        x_img, y_img = lensEquationSolver.image_position_from_source(
            kwargs_lens=kwargs_lens, sourcePos_x=x_source, sourcePos_y=y_source
        )
        x_img2, y_img2 = lensEquationSolver.image_position_from_source(
            kwargs_lens=kwargs_lens, sourcePos_x=x_source2, sourcePos_y=y_source2
        )
        print("image positions are: ", x_img, y_img)
        mag_inf = lensModel.magnification(x_img, y_img, kwargs_lens)
        mag_inf2 = lensModel.magnification(x_img2, y_img2, kwargs_lens)
        print("point source magnification: ", mag_inf)

        source_size_arcsec = 0.001
        window_size = 0.1
        grid_number = 100
        print("source size in arcsec: ", source_size_arcsec)
        mag_finite = lensModelExtensions.magnification_finite(
            x_pos=x_img,
            y_pos=y_img,
            kwargs_lens=kwargs_lens,
            source_sigma=source_size_arcsec,
            window_size=window_size,
            grid_number=grid_number,
        )
        flux_ratios = mag_finite[1:] / mag_finite[0]
        flux_ratio_errors = [0.1, 0.1, 0.1]
        flux_ratio_cov = np.diag([0.1, 0.1, 0.1]) ** 2

        flux_ratios2 = [
            np.array(mag_inf[1:] / mag_inf[0]),
            np.array(mag_inf2[1:] / mag_inf2[0]),
        ]
        flux_ratio_errors2 = [
            np.ones(len(flux_ratios2[0])),
            np.ones(len(flux_ratios2[1])),
        ]

        self.flux_likelihood = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios,
            flux_ratio_errors=flux_ratio_errors,
            source_type="GAUSSIAN",
            window_size=window_size,
            grid_number=grid_number,
        )

        self.flux_likelihood_inf = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios,
            flux_ratio_errors=flux_ratio_errors,
            source_type="INF",
            window_size=window_size,
            grid_number=grid_number,
        )
        self.flux_likelihood_inf_cov = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios,
            flux_ratio_errors=flux_ratio_cov,
            source_type="INF",
            window_size=window_size,
            grid_number=grid_number,
        )
        self.kwargs_cosmo = {"source_size": source_size_arcsec}
        self.x_img, self.y_img = [x_img], [y_img]
        self.x_img2, self.y_img2 = [x_img, x_img2], [y_img, y_img2]
        self.kwargs_lens = kwargs_lens

        self.flux_likelihood_inf2 = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios2,
            flux_ratio_errors=flux_ratio_errors2,
            source_type="INF",
            num_point_sources=2,
        )

    def test__logL(self):
        logL = self.flux_likelihood.logL(
            self.x_img, self.y_img, self.kwargs_lens, kwargs_special=self.kwargs_cosmo
        )
        assert logL == 0

        logL_inf = self.flux_likelihood_inf.logL(
            self.x_img, self.y_img, self.kwargs_lens, {}
        )
        npt.assert_almost_equal(logL_inf, 0, decimal=4)

    def test_logL(self):
        lensModel = LensModel(lens_model_list=[])
        flux_ratios_init = np.array([1.0, 1.0, 1.0])
        flux_ratio_errors = np.array([1.0, 1.0, 1.0])
        flux_likelihood = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios_init,
            flux_ratio_errors=flux_ratio_errors,
        )

        flux_ratios = np.array([0, 1, np.nan])
        logL = flux_likelihood._logL(flux_ratios)
        assert logL == -(10**15)

        flux_likelihood = FluxRatioLikelihood(
            lens_model_class=lensModel,
            flux_ratios=flux_ratios_init,
            flux_ratio_errors=np.array([0.0, 1.0, 1.0]),
        )
        flux_ratios = np.array([1.0, 1.0, 1.0])
        logL = flux_likelihood._logL(flux_ratios)
        assert logL == -(10**15)

    def test_numimgs(self):
        # Test with a different number of images
        logL = self.flux_likelihood.logL(
            [self.x_img[0][:-1]],
            [self.y_img[0][:-1]],
            self.kwargs_lens,
            kwargs_special=self.kwargs_cosmo,
        )
        assert logL == -(10**15)

    def test_covmatrix(self):
        # Test with a different number of images
        logL = self.flux_likelihood_inf_cov.logL(
            self.x_img, self.y_img, self.kwargs_lens, kwargs_special=self.kwargs_cosmo
        )
        npt.assert_almost_equal(logL, 0, decimal=8)

    def test_two_point_sources(self):
        """Same test but with two point sources."""
        logL = self.flux_likelihood_inf2.logL(
            self.x_img2, self.y_img2, self.kwargs_lens, kwargs_special={}
        )
        # no numerical test, just checking whether it can loop through
        npt.assert_almost_equal(logL, -2.7854474307328996, decimal=0)


if __name__ == "__main__":
    pytest.main()

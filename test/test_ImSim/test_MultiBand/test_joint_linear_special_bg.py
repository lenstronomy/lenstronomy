__author__ = "sibirrer"

import numpy.testing as npt
import numpy as np
import pytest

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.MultiBand.joint_linear_vary_background import JointLinear_VaryBG
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


class TestJointLinear_VaryBG(object):
    """Tests JointLinear_VaryBG, which extends JointLinear with a per-band constant
    background as an additional free linear parameter."""

    def setup_method(self):
        sigma_bkg = 0.05
        exp_time = 100
        numPix = 100
        deltaPix = 0.05
        fwhm = 0.5

        kwargs_data = sim_util.data_configure_simple(
            numPix, deltaPix, exp_time, sigma_bkg
        )
        data_class = ImageData(**kwargs_data)
        kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "truncation": 5}
        psf_class = PSF(**kwargs_psf)

        kwargs_shear = {"gamma1": 0.01, "gamma2": 0.01}
        phi, q = 0.2, 0.8
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_spemd = {
            "theta_E": 1.0,
            "gamma": 1.8,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_model_list = ["SPEP", "SHEAR"]
        self.kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)

        kwargs_sersic = {
            "amp": 1.0,
            "R_sersic": 0.1,
            "n_sersic": 2,
            "center_x": 0,
            "center_y": 0,
        }
        phi, q = 0.2, 0.9
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_sersic_ellipse = {
            "amp": 1.0,
            "R_sersic": 0.6,
            "n_sersic": 7,
            "center_x": 0,
            "center_y": 0,
            "e1": e1,
            "e2": e2,
        }

        lens_light_model_list = ["SERSIC"]
        self.kwargs_lens_light = [kwargs_sersic]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        source_model_list = ["SERSIC_ELLIPSE"]
        self.kwargs_source = [kwargs_sersic_ellipse]
        source_model_class = LightModel(light_model_list=source_model_list)
        self.kwargs_ps = [{"ra_source": 0.0001, "dec_source": 0.0, "source_amp": 1.0}]
        point_source_class = PointSource(
            point_source_type_list=["SOURCE_POSITION"], fixed_magnification_list=[True]
        )
        kwargs_numerics = {"supersampling_factor": 2, "supersampling_convolution": True}
        imageModel = ImageModel(
            data_class,
            psf_class,
            lens_model_class,
            source_model_class,
            lens_light_model_class,
            point_source_class,
            kwargs_numerics=kwargs_numerics,
        )
        # Simulate noiseless image, then add per-band background before noise so
        # that the Poisson noise model is self-consistent with the data (chi2 ~ 1).
        image_noiseless = sim_util.simulate_simple(
            imageModel,
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            no_noise=True,
        )

        self.num_bands = 2
        self.bg_band0 = 0.3
        self.bg_band1 = 0.7

        def make_band_image(noiseless, bg):
            img = noiseless + bg
            return (
                img
                + image_util.add_poisson(img, exp_time)
                + image_util.add_background(img, sigma_bkg)
            )

        image_sim_band0 = make_band_image(image_noiseless, self.bg_band0)
        image_sim_band1 = make_band_image(image_noiseless, self.bg_band1)

        data_class.update_data(image_sim_band0)
        kwargs_data["image_data"] = image_sim_band0

        # flux_scaling mimics per-exposure relative flux calibration (as in the
        # analysis script: multi_band_list[j][0]['flux_scaling'] = flux_scaling[j])
        kwargs_data_band0 = dict(
            kwargs_data, image_data=image_sim_band0, flux_scaling=1.0
        )
        kwargs_data_band1 = dict(
            kwargs_data, image_data=image_sim_band1, flux_scaling=0.8
        )
        multi_band_list = [
            [kwargs_data_band0, kwargs_psf, kwargs_numerics],
            [kwargs_data_band1, kwargs_psf, kwargs_numerics],
        ]
        kwargs_model = {
            "lens_model_list": lens_model_list,
            "source_light_model_list": source_model_list,
            "point_source_model_list": ["SOURCE_POSITION"],
            "fixed_magnification_list": [True],
            "lens_light_model_list": lens_light_model_list,
        }
        self.imageModel = JointLinear_VaryBG(multi_band_list, kwargs_model)
        # 3 joint linear params (source + lens light + point source) + 1 bg per band
        self.num_joint_params = 3
        self.num_params_total = self.num_joint_params + self.num_bands

    def test_linear_response(self):
        """Response matrix should have one extra row per band compared to
        JointLinear."""
        A = self.imageModel.linear_response_matrix(
            kwargs_lens=self.kwargs_lens,
            kwargs_source=self.kwargs_source,
            kwargs_lens_light=self.kwargs_lens_light,
            kwargs_ps=self.kwargs_ps,
        )
        nx, ny = np.shape(A)
        assert nx == self.num_params_total
        assert ny == 100**2 * self.num_bands

    def test_image_linear_solve(self):
        """image_linear_solve should return a single cov_param matrix and a flat param
        array (not per-band lists), with length = num_joint_params + num_bands."""
        wls_list, model_error_list, cov_param, param = (
            self.imageModel.image_linear_solve(
                self.kwargs_lens,
                self.kwargs_source,
                self.kwargs_lens_light,
                self.kwargs_ps,
                inv_bool=True,
            )
        )
        # wls_list has one reconstructed image per band
        assert len(wls_list) == self.num_bands
        # cov_param is a single 2D matrix, not a per-band list
        assert cov_param.ndim == 2
        assert cov_param.shape == (self.num_params_total, self.num_params_total)
        # param is a flat 1D array
        assert param.ndim == 1
        assert len(param) == self.num_params_total

    def test_background_params(self):
        """The per-band background parameters (last num_bands entries in param) should
        recover the known constant backgrounds injected into each band's data."""
        _, _, _, param = self.imageModel.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=True,
        )
        bg_params = param[self.num_joint_params :]
        assert len(bg_params) == self.num_bands
        npt.assert_almost_equal(bg_params[0], self.bg_band0, decimal=1)
        npt.assert_almost_equal(bg_params[1], self.bg_band1, decimal=1)

    def test_likelihood_data_given_model(self):
        logL, param = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
        )
        chi2_reduced = logL * 2 / self.imageModel.num_data_evaluate
        npt.assert_almost_equal(chi2_reduced, -1, 1)

    def test_likelihood_source_marg(self):
        """Test source_marg=True branch, which computes the marginalization constant
        from the covariance matrix."""
        logL_marg, param = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=True,
        )
        assert np.isfinite(logL_marg)

    def test_likelihood_check_positive_flux(self):
        """Test check_positive_flux=True branch, which penalises negative fluxes."""
        logL, param = self.imageModel.likelihood_data_given_model(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            source_marg=False,
            check_positive_flux=True,
        )
        assert np.isfinite(logL)

    def test_update_linear_kwargs(self):
        """Test that update_linear_kwargs runs without error and returns updated
        kwargs."""
        _, _, _, param = self.imageModel.image_linear_solve(
            self.kwargs_lens,
            self.kwargs_source,
            self.kwargs_lens_light,
            self.kwargs_ps,
            inv_bool=False,
        )
        kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps = (
            self.imageModel.update_linear_kwargs(
                param,
                0,
                self.kwargs_lens,
                self.kwargs_source,
                self.kwargs_lens_light,
                self.kwargs_ps,
            )
        )
        assert kwargs_source is not None


if __name__ == "__main__":
    pytest.main()

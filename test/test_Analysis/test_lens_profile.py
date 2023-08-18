import numpy as np
import numpy.testing as npt
import pytest
import unittest

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa
import lenstronomy.Util.param_util as param_util


class TestLensProfileAnalysis(object):
    def setup_method(self):
        pass

    def test_profile_slope(self):
        lens_model = LensProfileAnalysis(LensModel(lens_model_list=["SPP"]))
        gamma_in = 2.0
        kwargs_lens = [
            {"theta_E": 1.0, "gamma": gamma_in, "center_x": 0, "center_y": 0}
        ]
        gamma_out = lens_model.profile_slope(kwargs_lens, radius=1)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)
        gamma_in = 1.7
        kwargs_lens = [
            {"theta_E": 1.0, "gamma": gamma_in, "center_x": 0, "center_y": 0}
        ]
        gamma_out = lens_model.profile_slope(kwargs_lens, radius=1)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        gamma_in = 2.5
        kwargs_lens = [
            {"theta_E": 1.0, "gamma": gamma_in, "center_x": 0, "center_y": 0}
        ]
        gamma_out = lens_model.profile_slope(kwargs_lens, radius=1)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

        lens_model = LensProfileAnalysis(LensModel(lens_model_list=["SPEP"]))
        gamma_in = 2.0
        phi, q = 0.34403343049704888, 0.89760957136967312
        e1, e2 = param_util.phi_q2_ellipticity(phi, q)
        kwargs_lens = [
            {
                "theta_E": 1.4516812130749424,
                "e1": e1,
                "e2": e2,
                "center_x": -0.04507598845306314,
                "center_y": 0.054491803177414651,
                "gamma": gamma_in,
            }
        ]
        gamma_out = lens_model.profile_slope(kwargs_lens, radius=1.45)
        npt.assert_array_almost_equal(gamma_out, gamma_in, decimal=3)

    def test_effective_einstein_radius(self):
        kwargs_lens = [{"theta_E": 1, "center_x": 0, "center_y": 0}]
        lensModel = LensProfileAnalysis(LensModel(lens_model_list=["SIS"]))
        ret = lensModel.effective_einstein_radius_grid(kwargs_lens, get_precision=True)

        assert len(ret) == 2
        npt.assert_almost_equal(ret[0], 1.0, decimal=2)
        theta_E = lensModel.effective_einstein_radius(
            kwargs_lens, r_min=1e-3, r_max=1e1, num_points=30
        )
        npt.assert_almost_equal(theta_E, 1, decimal=3)

        kwargs_lens_bad = [{"theta_E": 100, "center_x": 0, "center_y": 0}]
        ret_nan, precision = lensModel.effective_einstein_radius_grid(
            kwargs_lens_bad, get_precision=True
        )
        assert np.isnan(ret_nan)
        theta_E = lensModel.effective_einstein_radius(
            kwargs_lens_bad, r_min=1e-3, r_max=1e1, num_points=30
        )
        assert np.isnan(theta_E)

        # test interpolated profile
        numPix = 101
        deltaPix = 0.02
        from lenstronomy.Util import util

        x_grid_interp, y_grid_interp = util.make_grid(numPix, deltaPix)
        from lenstronomy.LensModel.Profiles.sis import SIS

        sis = SIS()
        center_x, center_y = 0.0, -0.0
        kwargs_SIS = {"theta_E": 1.0, "center_x": center_x, "center_y": center_y}
        f_ = sis.function(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_x, f_y = sis.derivatives(x_grid_interp, y_grid_interp, **kwargs_SIS)
        f_xx, f_xy, f_yx, f_yy = sis.hessian(x_grid_interp, y_grid_interp, **kwargs_SIS)
        x_axes, y_axes = util.get_axes(x_grid_interp, y_grid_interp)
        kwargs_interpol = [
            {
                "grid_interp_x": x_axes,
                "grid_interp_y": y_axes,
                "f_": util.array2image(f_),
                "f_x": util.array2image(f_x),
                "f_y": util.array2image(f_y),
                "f_xx": util.array2image(f_xx),
                "f_xy": util.array2image(f_xy),
                "f_yy": util.array2image(f_yy),
            }
        ]
        lensModel = LensProfileAnalysis(LensModel(lens_model_list=["INTERPOL"]))
        theta_E_return = lensModel.effective_einstein_radius_grid(
            kwargs_interpol,
            get_precision=False,
            verbose=True,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(theta_E_return, 1, decimal=2)

        # sub-critical mass profile
        lensModel = LensProfileAnalysis(LensModel(lens_model_list=["NFW"]))
        kwargs_nfw = [{"Rs": 1, "alpha_Rs": 0.2, "center_x": 0, "center_y": 0}]
        theta_E_subcrit = lensModel.effective_einstein_radius_grid(
            kwargs_nfw, get_precision=False
        )
        assert np.isnan(theta_E_subcrit)
        theta_E_subcrit, _ = lensModel.effective_einstein_radius_grid(
            kwargs_nfw, get_precision=True
        )
        assert np.isnan(theta_E_subcrit)

    def test_external_lensing_effect(self):
        lens_model_list = ["SHEAR"]
        kwargs_lens = [{"gamma1": 0.1, "gamma2": 0.01}]
        lensModel = LensProfileAnalysis(LensModel(lens_model_list))
        alpha0_x, alpha0_y, kappa_ext, shear1, shear2 = lensModel.local_lensing_effect(
            kwargs_lens, model_list_bool=[0]
        )
        print(alpha0_x, alpha0_y, kappa_ext, shear1, shear2)
        assert alpha0_x == 0
        assert alpha0_y == 0
        assert shear1 == 0.1
        assert shear2 == 0.01
        assert kappa_ext == 0

    def test_multi_gaussian_lens(self):
        kwargs_options = {"lens_model_list": ["SPEP"]}
        lensModel = LensModel(**kwargs_options)
        lensAnalysis = LensProfileAnalysis(lens_model=lensModel)
        e1, e2 = param_util.phi_q2_ellipticity(0, 0.9)
        kwargs_lens = [
            {
                "gamma": 1.8,
                "theta_E": 0.6,
                "e1": e1,
                "e2": e2,
                "center_x": 0.5,
                "center_y": -0.1,
            }
        ]
        amplitudes, sigmas, center_x, center_y = lensAnalysis.multi_gaussian_lens(
            kwargs_lens, n_comp=20
        )
        model = MultiGaussianKappa()
        x = np.logspace(-2, 0.5, 10) + 0.5
        y = np.zeros_like(x) - 0.1
        f_xx, fxy, fyx, f_yy = model.hessian(
            x, y, amplitudes, sigmas, center_x=0.5, center_y=-0.1
        )
        kappa_mge = (f_xx + f_yy) / 2
        kappa_true = lensAnalysis._lens_model.kappa(x, y, kwargs_lens)
        print(kappa_true / kappa_mge)
        for i in range(len(x)):
            npt.assert_almost_equal(kappa_mge[i] / kappa_true[i], 1, decimal=1)

    def test_mass_fraction_within_radius(self):
        center_x, center_y = 0.5, -1
        theta_E = 1.1
        kwargs_lens = [{"theta_E": 1.1, "center_x": center_x, "center_y": center_y}]
        lensModel = LensModel(**{"lens_model_list": ["SIS"]})
        lensAnalysis = LensProfileAnalysis(lens_model=lensModel)
        kappa_mean_list = lensAnalysis.mass_fraction_within_radius(
            kwargs_lens, center_x, center_y, theta_E, numPix=100
        )
        npt.assert_almost_equal(kappa_mean_list[0], 1, 2)

    def test_lens_center(self):
        center_x, center_y = 0.43, -0.67
        kwargs_lens = [{"theta_E": 1, "center_x": center_x, "center_y": center_y}]
        lensModel = LensModel(**{"lens_model_list": ["SIS"]})
        profileAnalysis = LensProfileAnalysis(lens_model=lensModel)
        center_x_out, center_y_out = profileAnalysis.convergence_peak(kwargs_lens)
        npt.assert_almost_equal(center_x_out, center_x, 2)
        npt.assert_almost_equal(center_y_out, center_y, 2)

    def test_mst_invariant_differential(self):
        # testing with a SPP profile
        lensModel = LensModel(**{"lens_model_list": ["SPP"]})
        profileAnalysis = LensProfileAnalysis(lens_model=lensModel)
        gamma_list = [2, 1.8, 2.2]
        theta_E_list = [1, 0.5, 2]
        for gamma in gamma_list:
            for theta_E in theta_E_list:
                kwargs_lens = [
                    {"theta_E": theta_E, "gamma": gamma, "center_x": 0, "center_y": 0}
                ]
                xi = profileAnalysis.mst_invariant_differential(
                    kwargs_lens,
                    radius=theta_E,
                    center_x=None,
                    center_y=None,
                    model_list_bool=None,
                    num_points=10,
                )

                xi_true = (gamma - 2) / theta_E
                npt.assert_almost_equal(xi, xi_true, decimal=3)

        # it should also work in the elliptical regime
        lensModel = LensModel(**{"lens_model_list": ["EPL"]})
        profileAnalysis = LensProfileAnalysis(lens_model=lensModel)
        gamma_list = [2, 1.8, 2.2]
        theta_E_list = [1, 0.5, 2]
        q_list = [1, 0.9, 0.8, 0.7]
        for gamma in gamma_list:
            for theta_E in theta_E_list:
                for q in q_list:
                    e1, e2 = param_util.phi_q2_ellipticity(phi=0, q=q)
                    kwargs_lens = [
                        {
                            "theta_E": theta_E,
                            "gamma": gamma,
                            "e1": e1,
                            "e2": e2,
                            "center_x": 0,
                            "center_y": 0,
                        }
                    ]

                    theta_E_prime = np.sqrt(2 * q / (1 + q**2)) * theta_E
                    xi = profileAnalysis.mst_invariant_differential(
                        kwargs_lens,
                        radius=theta_E_prime,
                        center_x=None,
                        center_y=None,
                        model_list_bool=None,
                        num_points=10,
                    )
                    print(theta_E, gamma, q, "test")
                    xi_true = (gamma - 2) / theta_E_prime
                    npt.assert_almost_equal(xi, xi_true, decimal=2)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            raise ValueError()


if __name__ == "__main__":
    pytest.main()

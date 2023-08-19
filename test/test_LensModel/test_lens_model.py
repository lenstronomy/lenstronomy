__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.Util.util import make_grid
import unittest


class TestLensModel(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.lensModel = LensModel(["GAUSSIAN"])
        self.kwargs = [
            {
                "amp": 1.0,
                "sigma_x": 2.0,
                "sigma_y": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]

    def test_init(self):
        lens_model_list = [
            "FLEXION",
            "SIS_TRUNCATED",
            "SERSIC",
            "SERSIC_ELLIPSE_KAPPA",
            "SERSIC_ELLIPSE_GAUSS_DEC",
            "NFW_ELLIPSE_GAUSS_DEC",
            "SERSIC_ELLIPSE_POTENTIAL",
            "CTNFW_GAUSS_DEC",
            "PJAFFE",
            "PJAFFE_ELLIPSE",
            "HERNQUIST_ELLIPSE",
            "INTERPOL",
            "INTERPOL_SCALED",
            "SHAPELETS_POLAR",
            "DIPOLE",
            "GAUSSIAN_ELLIPSE_KAPPA",
            "GAUSSIAN_ELLIPSE_POTENTIAL",
            "MULTI_GAUSSIAN_KAPPA",
            "MULTI_GAUSSIAN_KAPPA_ELLIPSE",
            "CHAMELEON",
            "DOUBLE_CHAMELEON",
        ]

        lensModel = LensModel(lens_model_list)
        assert len(lensModel.lens_model_list) == len(lens_model_list)

        lens_model_list = ["NFW"]
        lensModel = LensModel(lens_model_list)
        x, y = 0.2, 1
        kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
        value = lensModel.potential(x, y, kwargs)
        nfw_interp = NFW(interpol=True)
        value_interp_lookup = nfw_interp.function(x, y, **kwargs[0])
        npt.assert_almost_equal(value, value_interp_lookup, decimal=4)

    def test_kappa(self):
        lensModel = LensModel(lens_model_list=["CONVERGENCE"])
        kappa_ext = 0.5
        kwargs = [{"kappa": kappa_ext}]
        output = lensModel.kappa(x=1.0, y=1.0, kwargs=kwargs)
        assert output == kappa_ext

    def test_potential(self):
        output = self.lensModel.potential(x=1.0, y=1.0, kwargs=self.kwargs)
        npt.assert_almost_equal(output, 0.77880078307140488 / (8 * np.pi), decimal=8)
        # assert output == 0.77880078307140488/(8*np.pi)

    def test_alpha(self):
        output1, output2 = self.lensModel.alpha(x=1.0, y=1.0, kwargs=self.kwargs)
        npt.assert_almost_equal(output1, -0.19470019576785122 / (8 * np.pi), decimal=8)
        npt.assert_almost_equal(output2, -0.19470019576785122 / (8 * np.pi), decimal=8)
        # assert output1 == -0.19470019576785122/(8*np.pi)
        # assert output2 == -0.19470019576785122/(8*np.pi)

        output1_diff, output2_diff = self.lensModel.alpha(
            x=1.0, y=1.0, kwargs=self.kwargs, diff=0.00001
        )
        npt.assert_almost_equal(output1_diff, output1, decimal=5)
        npt.assert_almost_equal(output2_diff, output2, decimal=5)

    def test_gamma(self):
        lensModel = LensModel(lens_model_list=["SHEAR"])
        gamma1, gamm2 = 0.1, -0.1
        kwargs = [{"gamma1": gamma1, "gamma2": gamm2}]
        e1_out, e2_out = lensModel.gamma(x=1.0, y=1.0, kwargs=kwargs)
        assert e1_out == gamma1
        assert e2_out == gamm2

        output1, output2 = self.lensModel.gamma(x=1.0, y=1.0, kwargs=self.kwargs)
        assert output1 == 0
        assert output2 == 0.048675048941962805 / (8 * np.pi)

    def test_magnification(self):
        output = self.lensModel.magnification(x=1.0, y=1.0, kwargs=self.kwargs)
        assert output == 0.98848384784633392

    def test_flexion(self):
        lensModel = LensModel(lens_model_list=["FLEXION"])
        g1, g2, g3, g4 = 0.01, 0.02, 0.03, 0.04
        kwargs = [{"g1": g1, "g2": g2, "g3": g3, "g4": g4}]
        f_xxx, f_xxy, f_xyy, f_yyy = lensModel.flexion(x=1.0, y=1.0, kwargs=kwargs)
        npt.assert_almost_equal(f_xxx, g1, decimal=8)
        npt.assert_almost_equal(f_xxy, g2, decimal=8)
        npt.assert_almost_equal(f_xyy, g3, decimal=8)
        npt.assert_almost_equal(f_yyy, g4, decimal=8)

    def test_ray_shooting(self):
        delta_x, delta_y = self.lensModel.ray_shooting(x=1.0, y=1.0, kwargs=self.kwargs)
        npt.assert_almost_equal(
            delta_x, 1 + 0.19470019576785122 / (8 * np.pi), decimal=8
        )
        npt.assert_almost_equal(
            delta_y, 1 + 0.19470019576785122 / (8 * np.pi), decimal=8
        )
        # assert delta_x == 1 + 0.19470019576785122/(8*np.pi)
        # assert delta_y == 1 + 0.19470019576785122/(8*np.pi)

    def test_arrival_time(self):
        z_lens = 0.5
        z_source = 1.5
        x_image, y_image = 1.0, 0.0
        lensModel = LensModel(
            lens_model_list=["SIS"],
            multi_plane=True,
            lens_redshift_list=[z_lens],
            z_source=z_source,
        )
        kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
        arrival_time_mp = lensModel.arrival_time(x_image, y_image, kwargs)
        lensModel_sp = LensModel(
            lens_model_list=["SIS"], z_source=z_source, z_lens=z_lens
        )
        arrival_time_sp = lensModel_sp.arrival_time(x_image, y_image, kwargs)
        npt.assert_almost_equal(arrival_time_sp, arrival_time_mp, decimal=8)

    def test_fermat_potential(self):
        z_lens = 0.5
        z_source = 1.5
        x_image, y_image = 1.0, 0.0
        lensModel = LensModel(
            lens_model_list=["SIS"],
            multi_plane=True,
            lens_redshift_list=[z_lens],
            z_lens=z_lens,
            z_source=z_source,
        )
        kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
        fermat_pot = lensModel.fermat_potential(x_image, y_image, kwargs)
        arrival_time = lensModel.arrival_time(x_image, y_image, kwargs)
        arrival_time_from_fermat_pot = lensModel._lensCosmo.time_delay_units(fermat_pot)
        npt.assert_almost_equal(arrival_time_from_fermat_pot, arrival_time, decimal=8)

    def test_curl(self):
        z_lens_list = [0.2, 0.8]
        z_source = 1.5
        lensModel = LensModel(
            lens_model_list=["SIS", "SIS"],
            multi_plane=True,
            lens_redshift_list=z_lens_list,
            z_source=z_source,
        )
        kwargs = [
            {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0},
            {"theta_E": 0.0, "center_x": 0.0, "center_y": 0.2},
        ]

        curl = lensModel.curl(x=1, y=1, kwargs=kwargs)
        assert curl == 0

        kwargs = [
            {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0},
            {"theta_E": 1.0, "center_x": 0.0, "center_y": 0.2},
        ]

        curl = lensModel.curl(x=1, y=1, kwargs=kwargs)
        assert curl != 0

    def test_hessian_differentials(self):
        """Routine to test the private numerical differentials, both cross and square
        methods in the infinitesimal regime."""
        lens_model = LensModel(lens_model_list=["SIS"])
        kwargs = [{"theta_E": 1, "center_x": 0.01, "center_y": 0}]
        x, y = make_grid(numPix=10, deltapix=0.2)
        diff = 0.0000001
        f_xx_sq, f_xy_sq, f_yx_sq, f_yy_sq = lens_model.hessian(
            x, y, kwargs, diff=diff, diff_method="square"
        )
        f_xx_cr, f_xy_cr, f_yx_cr, f_yy_cr = lens_model.hessian(
            x, y, kwargs, diff=diff, diff_method="cross"
        )
        f_xx, f_xy, f_yx, f_yy = lens_model.hessian(x, y, kwargs, diff=None)
        npt.assert_almost_equal(f_xx_cr, f_xx, decimal=5)
        npt.assert_almost_equal(f_xy_cr, f_xy, decimal=5)
        npt.assert_almost_equal(f_yx_cr, f_yx, decimal=5)
        npt.assert_almost_equal(f_yy_cr, f_yy, decimal=5)

        npt.assert_almost_equal(f_xx_sq, f_xx, decimal=5)
        npt.assert_almost_equal(f_xy_sq, f_xy, decimal=5)
        npt.assert_almost_equal(f_yx_sq, f_yx, decimal=5)
        npt.assert_almost_equal(f_yy_sq, f_yy, decimal=5)


class TestRaise(unittest.TestCase):
    def test_raise(self):
        with self.assertRaises(ValueError):
            kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
            lensModel = LensModel(
                ["NFW"], multi_plane=True, lens_redshift_list=[1], z_source=2
            )
            f_x, f_y = lensModel.alpha(1, 1, kwargs, diff=0.0001)
        with self.assertRaises(ValueError):
            lensModel = LensModel(["NFW"], multi_plane=True, lens_redshift_list=[1])
        with self.assertRaises(ValueError):
            kwargs = [{"alpha_Rs": 1, "Rs": 0.5, "center_x": 0, "center_y": 0}]
            lensModel = LensModel(["NFW"], multi_plane=False)
            t_arrival = lensModel.arrival_time(1, 1, kwargs)
        with self.assertRaises(ValueError):
            z_lens = 0.5
            z_source = 1.5
            x_image, y_image = 1.0, 0.0
            lensModel = LensModel(
                lens_model_list=["SIS"],
                multi_plane=True,
                lens_redshift_list=[z_lens],
                z_source=z_source,
            )
            kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
            fermat_pot = lensModel.fermat_potential(x_image, y_image, kwargs)
        with self.assertRaises(ValueError):
            lens_model = LensModel(lens_model_list=["SIS"])
            kwargs = [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}]
            lens_model.hessian(0, 0, kwargs, diff=0.001, diff_method="bad")
        with self.assertRaises(ValueError):
            lens_model = LensModel(lens_model_list=["LOS", "LOS_MINIMAL"])
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["LOS", "EPL", "NFW"], multi_plane=True, z_source=1.0
            )
        with self.assertRaises(ValueError):
            lens_model = LensModel(
                lens_model_list=["LOS_MINIMAL", "SIS", "GAUSSIAN"],
                multi_plane=True,
                z_source=1.0,
            )


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

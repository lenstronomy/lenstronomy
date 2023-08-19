__author__ = "sibirrer"

import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.curved_arc_sis_mst import CurvedArcSISMST
from lenstronomy.LensModel.Profiles.sis import SIS
from lenstronomy.LensModel.Profiles.convergence import Convergence
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions


class TestCurvedArcSISMST(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.model = CurvedArcSISMST()
        self.sis = SIS()
        self.mst = Convergence()

    def test_spp2stretch(self):
        center_x, center_y = 1, 1
        theta_E = 1
        kappa = 0.1
        center_x_spp, center_y_spp = 0.0, 0

        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.sis_mst2stretch(
            theta_E, kappa, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            kappa_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2sis_mst(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(kappa_new, kappa, decimal=8)

        center_x, center_y = -1, 1
        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.sis_mst2stretch(
            theta_E, kappa, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            kappa_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2sis_mst(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(kappa_new, kappa, decimal=8)

        center_x, center_y = 0, 0.5
        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.sis_mst2stretch(
            theta_E, kappa, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            kappa_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2sis_mst(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(kappa_new, kappa, decimal=8)

        center_x, center_y = 0, -1.5
        (
            tangential_stretch,
            radial_stretch,
            r_curvature,
            direction,
        ) = self.model.sis_mst2stretch(
            theta_E, kappa, center_x_spp, center_y_spp, center_x, center_y
        )
        print(tangential_stretch, radial_stretch, r_curvature, direction)
        (
            theta_E_new,
            kappa_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2sis_mst(
            tangential_stretch,
            radial_stretch,
            r_curvature,
            direction,
            center_x,
            center_y,
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(kappa_new, kappa, decimal=8)

    def test_function(self):
        center_x, center_y = 0.0, 0.0
        x, y = 1, 1
        radial_stretch = 1
        output = self.model.function(
            x,
            y,
            tangential_stretch=2,
            radial_stretch=radial_stretch,
            curvature=1.0 / 2,
            direction=0,
            center_x=center_x,
            center_y=center_y,
        )
        theta_E, kappa_ext, center_x_sis, center_y_sis = self.model.stretch2sis_mst(
            tangential_stretch=2,
            radial_stretch=radial_stretch,
            curvature=1.0 / 2,
            direction=0,
            center_x=center_x,
            center_y=center_y,
        )
        f_sis_out = self.sis.function(
            1, 1, theta_E, center_x_sis, center_y_sis
        )  # - self.sis.function(0, 0, theta_E, center_x_sis, center_y_sis)
        alpha_x, alpha_y = self.sis.derivatives(
            center_x, center_y, theta_E, center_x_sis, center_y_sis
        )
        f_sis_0_out = alpha_x * (x - center_x) + alpha_y * (y - center_y)

        f_mst_out = self.mst.function(x, y, kappa_ext, ra_0=center_x, dec_0=center_y)
        lambda_mst = 1.0 / radial_stretch
        f_out = lambda_mst * (f_sis_out - f_sis_0_out) + f_mst_out
        npt.assert_almost_equal(output, f_out, decimal=8)

    def test_derivatives(self):
        tangential_stretch = 5
        radial_stretch = 1
        curvature = 1.0 / 10
        direction = 0.3
        center_x = 0
        center_y = 0
        x, y = 1, 1
        theta_E, kappa, center_x_spp, center_y_spp = self.model.stretch2sis_mst(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        f_x_sis, f_y_sis = self.sis.derivatives(
            x, y, theta_E, center_x_spp, center_y_spp
        )
        f_x_mst, f_y_mst = self.mst.derivatives(
            x, y, kappa, ra_0=center_x, dec_0=center_y
        )
        f_x0, f_y0 = self.sis.derivatives(
            center_x, center_y, theta_E, center_x_spp, center_y_spp
        )
        f_x_new, f_y_new = self.model.derivatives(
            x,
            y,
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
            center_x,
            center_y,
        )
        npt.assert_almost_equal(f_x_new, f_x_sis + f_x_mst - f_x0, decimal=8)
        npt.assert_almost_equal(f_y_new, f_y_sis + f_y_mst - f_y0, decimal=8)

    def test_hessian(self):
        lens = LensModel(lens_model_list=["CURVED_ARC_SIS_MST"])
        center_x, center_y = 0, 0
        tangential_stretch = 10
        radial_stretch = 1
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

        center_x, center_y = 2, 3
        tangential_stretch = 10
        radial_stretch = 1
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

        center_x, center_y = 0, 0
        tangential_stretch = 5
        radial_stretch = 1.2
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

        center_x, center_y = 0, 0
        tangential_stretch = 3
        radial_stretch = -1
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        print(tangential_stretch, radial_stretch, "stretches")
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

        center_x, center_y = 0, 0
        tangential_stretch = -3
        radial_stretch = -1
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

        center_x, center_y = 0, 0
        tangential_stretch = 10.4
        radial_stretch = 0.6
        kwargs_lens = [
            {
                "tangential_stretch": tangential_stretch,
                "radial_stretch": radial_stretch,
                "curvature": 1.0 / 10.5,
                "direction": 0.0,
                "center_x": center_x,
                "center_y": center_y,
            }
        ]
        mag = lens.magnification(center_x, center_y, kwargs=kwargs_lens)
        npt.assert_almost_equal(mag, tangential_stretch * radial_stretch, decimal=8)

    def test_curved_arc_recovery(self):
        """Test whether the curved arc parameters are satisfied in differential form."""

        ext = LensModelExtensions(LensModel(lens_model_list=["CURVED_ARC_SIS_MST"]))
        center_x, center_y = (
            1,
            1.0,
        )  # test works except at (0,0) where the direction angle is not well defined
        tangential_stretch = 10.0
        radial_stretch = 1.2
        curvature, direction = 0.02, 0.5
        kwargs_lens = {
            "tangential_stretch": tangential_stretch,
            "radial_stretch": radial_stretch,
            "curvature": curvature,
            "direction": direction,
            "center_x": center_x,
            "center_y": center_y,
        }

        self._test_curved_arc_recovery(kwargs_lens)

    def _test_curved_arc_recovery(self, kwargs_arc_init):
        ext = LensModelExtensions(LensModel(lens_model_list=["CURVED_ARC_SIS_MST"]))
        center_x, center_y = kwargs_arc_init["center_x"], kwargs_arc_init["center_y"]
        kwargs_arc = ext.curved_arc_estimate(center_x, center_y, [kwargs_arc_init])
        (
            lambda_rad,
            lambda_tan,
            orientation_angle,
            dlambda_tan_dtan,
            dlambda_tan_drad,
            dlambda_rad_drad,
            dlambda_rad_dtan,
            dphi_tan_dtan,
            dphi_tan_drad,
            dphi_rad_drad,
            dphi_rad_dtan,
        ) = ext.radial_tangential_differentials(center_x, center_y, [kwargs_arc_init])
        npt.assert_almost_equal(
            kwargs_arc["tangential_stretch"],
            kwargs_arc_init["tangential_stretch"],
            decimal=3,
        )
        npt.assert_almost_equal(
            kwargs_arc["radial_stretch"], kwargs_arc_init["radial_stretch"], decimal=3
        )
        npt.assert_almost_equal(
            kwargs_arc["curvature"], kwargs_arc_init["curvature"], decimal=3
        )
        npt.assert_almost_equal(dphi_tan_dtan, kwargs_arc_init["curvature"], decimal=3)
        npt.assert_almost_equal(
            kwargs_arc["direction"], kwargs_arc_init["direction"], decimal=3
        )
        npt.assert_almost_equal(dlambda_tan_dtan, 0, decimal=3)


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

__author__ = "sibirrer"

import numpy as np
import numpy.testing as npt
import pytest
from lenstronomy.LensModel.Profiles.curved_arc_spp import CurvedArcSPP
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.lens_model import LensModel


class TestCurvedArc(object):
    """Tests the source model routines."""

    def setup_method(self):
        self.model = CurvedArcSPP()
        self.spp = SPP()

    def test_spp2stretch(self):
        center_x, center_y = 1, 1
        theta_E = 1
        gamma = 1.9
        center_x_spp, center_y_spp = 0.0, 0

        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.spp2stretch(
            theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            gamma_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2spp(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(gamma_new, gamma, decimal=8)

        center_x, center_y = -1, 1
        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.spp2stretch(
            theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            gamma_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2spp(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(gamma_new, gamma, decimal=8)

        center_x, center_y = 0, 0.5
        (
            tangential_stretch,
            radial_stretch,
            curvature,
            direction,
        ) = self.model.spp2stretch(
            theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y
        )
        (
            theta_E_new,
            gamma_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2spp(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        npt.assert_almost_equal(center_x_spp_new, center_x_spp, decimal=8)
        npt.assert_almost_equal(center_y_spp_new, center_y_spp, decimal=8)
        npt.assert_almost_equal(theta_E_new, theta_E, decimal=8)
        npt.assert_almost_equal(gamma_new, gamma, decimal=8)

        center_x, center_y = 0, -1.5
        (
            tangential_stretch,
            radial_stretch,
            r_curvature,
            direction,
        ) = self.model.spp2stretch(
            theta_E, gamma, center_x_spp, center_y_spp, center_x, center_y
        )
        print(tangential_stretch, radial_stretch, r_curvature, direction)
        (
            theta_E_new,
            gamma_new,
            center_x_spp_new,
            center_y_spp_new,
        ) = self.model.stretch2spp(
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
        npt.assert_almost_equal(gamma_new, gamma, decimal=8)

    def test_function(self):
        center_x, center_y = 0.0, 0.0
        x, y = 1, 1
        output = self.model.function(
            x,
            y,
            tangential_stretch=2,
            radial_stretch=1,
            curvature=1.0 / 2,
            direction=0,
            center_x=0,
            center_y=0,
        )
        theta_E, gamma, center_x_spp, center_y_spp = self.model.stretch2spp(
            tangential_stretch=2,
            radial_stretch=1,
            curvature=1.0 / 2,
            direction=0,
            center_x=0,
            center_y=0,
        )
        out_spp = self.spp.function(1, 1, theta_E, gamma, center_x_spp, center_y_spp)
        alpha_x, alpha_y = self.spp.derivatives(
            center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp
        )
        f_0 = alpha_x * (x - center_x) + alpha_y * (y - center_y)

        npt.assert_almost_equal(output, out_spp - f_0, decimal=8)

    def test_derivatives(self):
        tangential_stretch = 5
        radial_stretch = 1
        curvature = 1.0 / 10
        direction = 0.3
        center_x = 0
        center_y = 0
        x, y = 1, 1
        theta_E, gamma, center_x_spp, center_y_spp = self.model.stretch2spp(
            tangential_stretch, radial_stretch, curvature, direction, center_x, center_y
        )
        f_x, f_y = self.spp.derivatives(
            x, y, theta_E, gamma, center_x_spp, center_y_spp
        )
        f_x0, f_y0 = self.spp.derivatives(
            center_x, center_y, theta_E, gamma, center_x_spp, center_y_spp
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
        npt.assert_almost_equal(f_x_new, f_x - f_x0, decimal=8)
        npt.assert_almost_equal(f_y_new, f_y - f_y0, decimal=8)

    def test_hessian(self):
        lens = LensModel(lens_model_list=["CURVED_ARC_SPP"])
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


if __name__ == "__main__":
    pytest.main("-k TestLensModel")

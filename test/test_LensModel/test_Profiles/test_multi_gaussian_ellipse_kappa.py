__author__ = "xinchzhu"


from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import (
    GaussianEllipseKappa,
)
from lenstronomy.LensModel.Profiles.multi_gaussian_ellipse_kappa import (
    MultiGaussianEllipseKappa,
)

import numpy as np
import numpy.testing as npt
import pytest


class TestGaussianEllipseKappa(object):
    """Test the Gaussian with Gaussian Ellipse Kappa."""

    """Including 2 options: fixed ellipticity (input one single value of each e1 and e2)
        /variable ellipticities (input lists for both e1 and e2 for different components)"""

    def setup_method(self):
        self.multi = MultiGaussianEllipseKappa()
        self.single = GaussianEllipseKappa()

    def test_function(self):
        ############
        # fixed ellipticity
        ############
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_ = self.multi.function(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        ############
        # variable ellipticities
        ############
        f_e = self.multi.function(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=[e1],
            e2=[e2],
            center_x=center_x,
            center_y=center_y,
        )
        f_single = self.single.function(
            x,
            y,
            amp=amp,
            sigma=sigma,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(f_, f_single, decimal=8)
        npt.assert_almost_equal(f_e, f_single, decimal=8)

    def test_derivatives(self):
        ############
        # fixed ellipticity
        ############
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_x, f_y = self.multi.derivatives(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        ############
        # variable ellipticities
        ############
        f_x_e, f_y_e = self.multi.derivatives(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=[e1],
            e2=[e2],
            center_x=center_x,
            center_y=center_y,
        )
        f_x_s, f_y_s = self.single.derivatives(
            x,
            y,
            amp=amp,
            sigma=sigma,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(f_x, f_x_s, decimal=8)
        npt.assert_almost_equal(f_y, f_y_s, decimal=8)
        npt.assert_almost_equal(f_x_e, f_x_s, decimal=8)
        npt.assert_almost_equal(f_y_e, f_y_s, decimal=8)

    def test_hessian(self):
        ############
        # fixed ellipticity
        ############
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_xx, f_xy, f_yx, f_yy = self.multi.hessian(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        ############
        # variable ellipticities
        ############
        f_xx_e, f_xy_e, f_yx_e, f_yy_e = self.multi.hessian(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=[e1],
            e2=[e2],
            center_x=center_x,
            center_y=center_y,
        )
        f_xx_s, f_xy_s, f_yx_s, f_yy_s = self.single.hessian(
            x,
            y,
            amp=amp,
            sigma=sigma,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(f_xx, f_xx_s, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_s, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_s, decimal=8)
        npt.assert_almost_equal(f_yx, f_yx_s, decimal=7)
        npt.assert_almost_equal(f_xx_e, f_xx_s, decimal=8)
        npt.assert_almost_equal(f_yy_e, f_yy_s, decimal=8)
        npt.assert_almost_equal(f_xy_e, f_xy_s, decimal=8)
        npt.assert_almost_equal(f_yx_e, f_yx_s, decimal=7)

    def test_density_2d(self):
        ############
        # fixed ellipticity
        ############
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_ = self.multi.density_2d(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        ############
        # variable ellipticities
        ############
        f_e = self.multi.density_2d(
            x,
            y,
            amp=[amp],
            sigma=[sigma],
            e1=[e1],
            e2=[e2],
            center_x=center_x,
            center_y=center_y,
        )
        f_single = self.single.density_2d(
            x,
            y,
            amp=amp,
            sigma=sigma,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
        )
        npt.assert_almost_equal(f_, f_single, decimal=8)
        npt.assert_almost_equal(f_e, f_single, decimal=8)


if __name__ == "__main__":
    pytest.main()

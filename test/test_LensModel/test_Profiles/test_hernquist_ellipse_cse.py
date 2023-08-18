__author__ = "sibirrer"


from lenstronomy.LensModel.Profiles.hernquist_ellipse_cse import HernquistEllipseCSE
from lenstronomy.LensModel.Profiles.hernquist import Hernquist

import numpy as np
import numpy.testing as npt
import pytest


class TestHernquistEllipseCSE(object):
    """
    tests the Gaussian methods
    """

    def setup_method(self):
        self.hernquist = Hernquist()
        self.hernquist_cse = HernquistEllipseCSE()

    def test_function(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 2, "Rs": 2, "center_x": 0, "center_y": 0}

        f_nfw = self.hernquist.function(x, y, **kwargs)
        f_cse = self.hernquist_cse.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_cse / f_nfw, 1, decimal=5)

    def test_derivatives(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_x_nfw, f_y_nfw = self.hernquist.derivatives(x, y, **kwargs)
        f_x_cse, f_y_cse = self.hernquist_cse.derivatives(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_x_cse, f_x_nfw, decimal=5)
        npt.assert_almost_equal(f_y_cse, f_y_nfw, decimal=5)

    def test_hessian(self):
        x = np.linspace(0.01, 5, 30)
        y = np.zeros_like(x)
        kwargs = {"sigma0": 0.5, "Rs": 2, "center_x": 0, "center_y": 0}

        f_xx_nfw, f_xy_nfw, f_yx_nfw, f_yy_nfw = self.hernquist.hessian(x, y, **kwargs)
        f_xx_cse, f_xy_cse, f_yx_cse, f_yy_cse = self.hernquist_cse.hessian(
            x, y, e1=0, e2=0, **kwargs
        )
        npt.assert_almost_equal(f_xx_cse / f_xx_nfw, 1, decimal=2)
        npt.assert_almost_equal(f_xy_cse, f_xy_nfw, decimal=5)
        npt.assert_almost_equal(f_yx_cse, f_yx_nfw, decimal=5)
        npt.assert_almost_equal(f_yy_cse, f_yy_nfw, decimal=5)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        m_3d_nfw = self.hernquist.mass_3d_lens(R, Rs, alpha_Rs)
        m_3d_cse = self.hernquist_cse.mass_3d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_3d_nfw, m_3d_cse, decimal=8)


if __name__ == "__main__":
    pytest.main()

__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE

import numpy as np
import numpy.testing as npt
import pytest


class TestNFWELLIPSE(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()
        self.nfw_cse = NFW_ELLIPSE_CSE(high_accuracy=True)
        self.nfw_cse_low_accuracy = NFW_ELLIPSE_CSE(high_accuracy=False)

    def test_function(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {'alpha_Rs': 2, 'Rs': 2, 'center_x': 0, 'center_y': 0}

        f_nfw = self.nfw.function(x, y, **kwargs)
        f_cse = self.nfw_cse.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_cse, f_nfw, decimal=5)
        f_cse_low = self.nfw_cse_low_accuracy.function(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_cse_low / f_nfw, 1, decimal=3)

    def test_derivatives(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0}

        f_x_nfw, f_y_nfw = self.nfw.derivatives(x, y, **kwargs)
        f_x_cse, f_y_cse = self.nfw_cse.derivatives(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_x_cse, f_x_nfw, decimal=5)
        npt.assert_almost_equal(f_y_cse, f_y_nfw, decimal=5)
        f_x_cse_low, f_y_cse_low = self.nfw_cse_low_accuracy.derivatives(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_x_cse_low / f_x_nfw, 1, decimal=2)
        npt.assert_almost_equal(f_y_cse_low,  f_y_nfw, decimal=2)

    def test_hessian(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0}

        f_xx_nfw, f_xy_nfw, f_yx_nfw, f_yy_nfw = self.nfw.hessian(x, y, **kwargs)
        f_xx_cse, f_xy_cse, f_yx_cse, f_yy_cse = self.nfw_cse.hessian(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_xx_cse, f_xx_nfw, decimal=5)
        npt.assert_almost_equal(f_xy_cse, f_xy_nfw, decimal=5)
        npt.assert_almost_equal(f_yx_cse, f_yx_nfw, decimal=5)
        npt.assert_almost_equal(f_yy_cse, f_yy_nfw, decimal=5)

        f_xx_cse, f_xy_cse, f_yx_cse, f_yy_cse = self.nfw_cse_low_accuracy.hessian(x, y, e1=0, e2=0, **kwargs)
        npt.assert_almost_equal(f_xx_cse / f_xx_nfw, 1, decimal=1)
        npt.assert_almost_equal(f_xy_cse, f_xy_nfw, decimal=5)
        npt.assert_almost_equal(f_yx_cse, f_yx_nfw, decimal=5)
        npt.assert_almost_equal(f_yy_cse / f_yy_nfw, 1, decimal=1)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        m_3d_nfw = self.nfw.mass_3d_lens(R, Rs, alpha_Rs)
        m_3d_cse = self.nfw_cse.mass_3d_lens(R, Rs, alpha_Rs)
        npt.assert_almost_equal(m_3d_nfw, m_3d_cse, decimal=8)


if __name__ == '__main__':
    pytest.main()

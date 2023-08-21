__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse_cse import NFW_ELLIPSE_CSE
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.lens_model import LensModel

import numpy as np
import numpy.testing as npt
import pytest


class TestNFWELLIPSE(object):
    """
    tests the Gaussian methods
    """
    def setup_method(self):
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

    def test_ellipticity(self):
        """
        test the definition of the ellipticity normalization (along major axis or product averaged axes)
        """
        x, y = np.linspace(start=0.001, stop=10, num=100), np.zeros(100)
        kwargs_round = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0, 'e1': 0.3, 'e2': 0}

        f_xx, f_xy, f_yx, f_yy = self.nfw_cse.hessian(x, y, **kwargs_round)
        kappa_round = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.nfw_cse.hessian(x, y, **kwargs)
        kappa_major = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.nfw_cse.hessian(y, x, **kwargs)
        kappa_minor = 1. / 2 * (f_xx + f_yy)

        npt.assert_almost_equal(np.sqrt(kappa_minor * kappa_major),kappa_round, decimal=2)

        # import matplotlib.pyplot as plt
        # plt.plot(x, kappa_round/kappa_round, ':', label='round', alpha=0.5)
        # plt.plot(x, kappa_major/kappa_round, ',-', label='major', alpha=0.5)
        # plt.plot(x, kappa_minor/kappa_round, '--', label='minor', alpha=0.5)
        # plt.plot(x, np.sqrt(kappa_minor * kappa_major)/kappa_round, '--', label='prod', alpha=0.5)
        # plt.plot(x, np.sqrt(kappa_minor**2 + kappa_major**2) / kappa_round / 2, '--', label='square', alpha=0.5)
        # plt.legend()
        # plt.show()
    def test_einstein_rad(self):
        """
         test that the Einstein radius doesn't change significantly with ellipticity
         """
        kwargs_round = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'center_x': 0, 'center_y': 0, 'e1': 0.3, 'e2': 0}
        LensMod = LensModel(['NFW_ELLIPSE_CSE'])
        LensAn = LensProfileAnalysis(LensMod)
        r_Ein_round = LensAn.effective_einstein_radius([kwargs_round])
        r_Ein_ell = LensAn.effective_einstein_radius([kwargs])
        npt.assert_almost_equal(r_Ein_round, r_Ein_ell, decimal=1)


if __name__ == '__main__':
    pytest.main()

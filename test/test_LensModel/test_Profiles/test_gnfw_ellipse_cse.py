__author__ = 'mgomer'

from lenstronomy.LensModel.Profiles.general_nfw import GNFW
from lenstronomy.LensModel.Profiles.gnfw_ellipse_cse import GNFW_ELLIPSE_CSE
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
        self.gnfw = GNFW()
        self.gnfw_cse = GNFW_ELLIPSE_CSE()

    def test_function(self):
        #Potential not implemented in GNFW to compare
        npt.assert_almost_equal(1, 1)

    def test_derivatives(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        for gamma_inner in [0.5,1.5]:
            for gamma_outer in [2.5,4]:
                kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0, 'center_y': 0}
                f_x_gnfw, f_y_gnfw = self.gnfw.derivatives(x, y, **kwargs)
                f_x_cse, f_y_cse = self.gnfw_cse.derivatives(x, y, e1=0, e2=0, **kwargs)
                npt.assert_almost_equal(f_x_cse, f_x_gnfw, decimal=5)
                npt.assert_almost_equal(f_y_cse, f_y_gnfw, decimal=5)

    def test_hessian(self):
        x = np.linspace(0.01, 2, 10)
        y = np.zeros_like(x)
        for gamma_inner in [0.5,1.5]:
            for gamma_outer in [2.5,4]:
                kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0, 'center_y': 0}
                f_xx_gnfw, f_xy_gnfw, f_yx_gnfw, f_yy_gnfw = self.gnfw.hessian(x, y, **kwargs)
                f_xx_cse, f_xy_cse, f_yx_cse, f_yy_cse = self.gnfw_cse.hessian(x, y, e1=0, e2=0, **kwargs)
                npt.assert_almost_equal(f_xx_cse, f_xx_gnfw, decimal=5)
                npt.assert_almost_equal(f_xy_cse, f_xy_gnfw, decimal=5)
                npt.assert_almost_equal(f_yx_cse, f_yx_gnfw, decimal=5)
                npt.assert_almost_equal(f_yy_cse, f_yy_gnfw, decimal=5)

    def test_mass_3d_lens(self):
        R = 1
        Rs = 3
        alpha_Rs = 1
        for gamma_inner in [0.5,1.5]:
            for gamma_outer in [2.5,4]:
                m_3d_gnfw = self.gnfw.mass_3d_lens(R, Rs, alpha_Rs, gamma_inner= gamma_inner, gamma_outer= gamma_outer)
                m_3d_cse = self.gnfw_cse.mass_3d_lens(R, Rs, alpha_Rs, gamma_inner= gamma_inner, gamma_outer= gamma_outer)
                npt.assert_almost_equal(m_3d_gnfw, m_3d_cse, decimal=8)

    def test_ellipticity(self):
        """
        test the definition of the ellipticity normalization (along major axis or product averaged axes)
        """
        x, y = np.linspace(start=0.001, stop=10, num=100), np.zeros(100)
        gamma_inner=1
        gamma_outer=3
        kwargs_round = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0,
                        'center_y': 0, 'e1': 0, 'e2': 0}
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0,
                  'center_y': 0, 'e1': 0.3, 'e2': 0}

        f_xx, f_xy, f_yx, f_yy = self.gnfw_cse.hessian(x, y, **kwargs_round)
        kappa_round = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.gnfw_cse.hessian(x, y, **kwargs)
        kappa_major = 1. / 2 * (f_xx + f_yy)

        f_xx, f_xy, f_yx, f_yy = self.gnfw_cse.hessian(y, x, **kwargs)
        kappa_minor = 1. / 2 * (f_xx + f_yy)

        npt.assert_almost_equal(np.sqrt(kappa_minor * kappa_major),kappa_round, decimal=2)

    def test_einstein_rad(self):
        """
         test that the Einstein radius doesn't change significantly with ellipticity
         """
        gamma_inner=1
        gamma_outer=3
        kwargs_round = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0, 'center_y': 0, 'e1': 0, 'e2': 0}
        kwargs = {'alpha_Rs': 0.5, 'Rs': 2, 'gamma_inner': gamma_inner, 'gamma_outer': gamma_outer, 'center_x': 0, 'center_y': 0, 'e1': 0.3, 'e2': 0}
        LensMod=LensModel(['GNFW_ELLIPSE_CSE'])
        LensAn=LensProfileAnalysis(LensMod)
        r_Ein_round=LensAn.effective_einstein_radius([kwargs_round])
        r_Ein_ell=LensAn.effective_einstein_radius([kwargs])
        npt.assert_almost_equal(r_Ein_round,r_Ein_ell,decimal=1)


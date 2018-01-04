__author__ = 'sibirrer'

import lenstronomy.Util.derivative_util as calc_util
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.sersic import Sersic as Sersic_light

import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = Sersic()
        self.sersic_light = Sersic_light()

    def test_function(self):
        x = 1
        y = 2
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values, 1.0272982586319199, decimal=10)

        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values[0], 0., decimal=10)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function(x, y, n_sersic, r_eff, k_eff)
        npt.assert_almost_equal(values[0], 1.0272982586319199, decimal=10)
        npt.assert_almost_equal(values[1], 1.3318743892966658, decimal=10)
        npt.assert_almost_equal(values[2], 1.584299393114988, decimal=10)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0.16556078301997193
        assert f_y[0] == 0.33112156603994386
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.derivatives(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.16556078301997193
        assert values[1][0] == 0.33112156603994386
        assert values[0][1] == 0.2772992378623737
        assert values[1][1] == 0.092433079287457892

    def test_differentails(self):
        x_, y_ = 1., 1
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        r = np.sqrt(x_**2 + y_**2)
        d_alpha_dr = self.sersic.d_alpha_dr(x_, y_, n_sersic, r_eff, k_eff)
        alpha = self.sersic.alpha_abs(x_, y_, n_sersic, r_eff, k_eff)

        f_xx_ = d_alpha_dr * calc_util.d_r_dx(x_, y_) * x_/r + alpha * calc_util.d_x_diffr_dx(x_, y_)
        f_yy_ = d_alpha_dr * calc_util.d_r_dy(x_, y_) * y_/r + alpha * calc_util.d_y_diffr_dy(x_, y_)
        f_xy_ = d_alpha_dr * calc_util.d_r_dy(x_, y_) * x_/r + alpha * calc_util.d_x_diffr_dy(x_, y_)

        f_xx = (d_alpha_dr/r - alpha/r**2) * y_**2/r + alpha/r
        f_yy = (d_alpha_dr/r - alpha/r**2) * x_**2/r + alpha/r
        f_xy = (d_alpha_dr/r - alpha/r**2) * x_*y_/r
        npt.assert_almost_equal(f_xx, f_xx_, decimal=10)
        npt.assert_almost_equal(f_yy, f_yy_, decimal=10)
        npt.assert_almost_equal(f_xy, f_xy_, decimal=10)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        f_xx, f_yy,f_xy = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert f_xx[0] == 0.1123170666045793
        npt.assert_almost_equal(f_yy[0], -0.047414082641598576, decimal=10)
        npt.assert_almost_equal(f_xy[0], -0.10648743283078525 , decimal=10)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        assert values[0][0] == 0.1123170666045793
        npt.assert_almost_equal(values[1][0], -0.047414082641598576, decimal=10)
        npt.assert_almost_equal(values[2][0], -0.10648743283078525 , decimal=10)
        npt.assert_almost_equal(values[0][1], -0.053273787681591328, decimal=10)
        npt.assert_almost_equal(values[1][1], 0.076243427402007985, decimal=10)
        npt.assert_almost_equal(values[2][1], -0.048568955656349749, decimal=10)

    def test_alpha_abs(self):
        x = 1.
        dr = 0.0000001
        n_sersic = 2.5
        r_eff = .5
        k_eff = 0.2
        alpha_abs = self.sersic.alpha_abs(x, 0, n_sersic, r_eff, k_eff)
        f_dr = self.sersic.function(x + dr, 0, n_sersic, r_eff, k_eff)
        f_ = self.sersic.function(x, 0, n_sersic, r_eff, k_eff)
        alpha_abs_num = -(f_dr - f_)/dr
        npt.assert_almost_equal(alpha_abs_num, alpha_abs, decimal=3)

    def test_dalpha_dr(self):
        x = 1.
        dr = 0.0000001
        n_sersic = 1.
        r_eff = .5
        k_eff = 0.2
        d_alpha_dr = self.sersic.d_alpha_dr(x, 0, n_sersic, r_eff, k_eff)
        alpha_dr = self.sersic.alpha_abs(x + dr, 0, n_sersic, r_eff, k_eff)
        alpha = self.sersic.alpha_abs(x, 0, n_sersic, r_eff, k_eff)
        d_alpha_dr_num = (alpha_dr - alpha)/dr
        npt.assert_almost_equal(d_alpha_dr, d_alpha_dr_num, decimal=3)

    def test_mag_sym(self):
        """

        :return:
        """
        r = 2.
        angle1 = 0.
        angle2 = 1.5
        x1 = r * np.cos(angle1)
        y1 = r * np.sin(angle1)

        x2 = r * np.cos(angle2)
        y2 = r * np.sin(angle2)
        n_sersic = 4.5
        r_eff = 2.5
        k_eff = 0.8
        f_xx1, f_yy1, f_xy1 = self.sersic.hessian(x1, y1, n_sersic, r_eff, k_eff)
        f_xx2, f_yy2, f_xy2 = self.sersic.hessian(x2, y2, n_sersic, r_eff, k_eff)
        kappa_1 = (f_xx1 + f_yy1) / 2
        kappa_2 = (f_xx2 + f_yy2) / 2
        npt.assert_almost_equal(kappa_1, kappa_2, decimal=10)
        A_1 = (1 - f_xx1) * (1 - f_yy1) - f_xy1**2
        A_2 = (1 - f_xx2) * (1 - f_yy2) - f_xy2 ** 2
        npt.assert_almost_equal(A_1, A_2, decimal=10)

    def test_convergernce(self):
        """
        test the convergence and compares it with the original Sersic profile
        :return:
        """
        x = np.array([0, 0, 0, 0, 0])
        y = np.array([0.5, 1, 1.5, 2, 2.5])
        n_sersic = 4.5
        r_eff = 2.5
        k_eff = 0.2
        f_xx, f_yy, f_xy = self.sersic.hessian(x, y, n_sersic, r_eff, k_eff)
        kappa = (f_xx + f_yy) / 2.
        assert kappa[0] > 0
        flux = self.sersic_light.function(x, y, I0_sersic=1., R_sersic=r_eff, n_sersic=n_sersic)
        flux /= flux[0]
        kappa /= kappa[0]
        npt.assert_almost_equal(flux[1], kappa[1], decimal=5)

    def test_sersic_util(self):
        n = 1.
        Re = 2.
        k, bn = self.sersic.k_bn(n, Re)
        Re_new = self.sersic.k_Re(n, k)
        assert Re == Re_new


if __name__ == '__main__':
    pytest.main()
__author__ = 'ajshajib'

from lenstronomy.LensModel.Profiles.sersic_ellipse_gauss_dec import SersicEllipseGaussDec
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic

import numpy as np
import numpy.testing as npt
import pytest


class TestGaussianKappaEllipse(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.sersic_gauss = SersicEllipseGaussDec()
        self.sersic_light = SersicElliptic()
        self.sersic_sphere = Sersic()

    def test_function(self):
        k_eff = 1
        R_sersic = 1
        n_sersic = 1
        e1 = 0.2
        e2 = 0.2
        center_x = 0.
        center_y = 0.

        diff = 1e-6

        n = 5
        xs = np.linspace(0.5 * R_sersic, 2 * R_sersic, n)
        ys = np.linspace(0.5 * R_sersic, 2 * R_sersic, n)

        for x, y in zip(xs, ys):
            func = self.sersic_gauss.function(x, y, n_sersic, R_sersic,
                                              k_eff, e1, e2, center_x, center_y)

            func_dx = self.sersic_gauss.function(x+diff, y, n_sersic, R_sersic,
                                                 k_eff, e1, e2, center_x, center_y)

            func_dy = self.sersic_gauss.function(x, y+diff, n_sersic,
                                                 R_sersic, k_eff, e1, e2, center_x,
                                                 center_y)

            f_x_num = (func_dx - func) / diff
            f_y_num = (func_dy - func) / diff

            f_x, f_y = self.sersic_gauss.derivatives(x, y, n_sersic, R_sersic,
                                                     k_eff, e1, e2, center_x,
                                                     center_y)

            npt.assert_almost_equal(f_x_num, f_x, decimal=4)
            npt.assert_almost_equal(f_y_num, f_y, decimal=4)

    def test_derivatives(self):
        k_eff = 1
        R_sersic = 1
        n_sersic = 1
        e1 = 5e-5
        e2 = 0.0
        center_x = 0.
        center_y = 0.

        n = 10
        x = np.linspace(0.5*R_sersic, 2*R_sersic, n)
        y = np.linspace(0.5*R_sersic, 2*R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_x_s, f_y_s = self.sersic_sphere.derivatives(X, Y, n_sersic,
                                                      R_sersic, k_eff,
                                                      center_x, center_y)
        f_x, f_y = self.sersic_gauss.derivatives(X, Y, n_sersic, R_sersic,
                                                 k_eff, e1, e2, center_x,
                                                 center_y)

        npt.assert_allclose(f_x, f_x_s, rtol=1e-3, atol=0)
        npt.assert_allclose(f_y, f_y_s, rtol=1e-3, atol=0)

        npt.assert_almost_equal(f_x, f_x_s, decimal=3)
        npt.assert_almost_equal(f_y, f_y_s, decimal=3)

    def test_hessian(self):
        k_eff = 1
        R_sersic = 1
        n_sersic = 1
        e1 = 5e-5
        e2 = 0.0
        center_x = 0.
        center_y = 0.

        n = 10
        x = np.linspace(0.5 * R_sersic, 2 * R_sersic, n)
        y = np.linspace(0.5 * R_sersic, 2 * R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_xx_s, f_yy_s, f_xy_s = self.sersic_sphere.hessian(X, Y, n_sersic,
                                                            R_sersic, k_eff,
                                                            center_x, center_y)
        f_xx, f_yy, f_xy = self.sersic_gauss.hessian(X, Y, n_sersic, R_sersic,
                                                     k_eff, e1, e2, center_x,
                                                     center_y)

        npt.assert_almost_equal(f_xx_s, f_xx, decimal=3)
        npt.assert_almost_equal(f_yy_s, f_yy, decimal=3)
        npt.assert_almost_equal(f_xy_s, f_xy, decimal=3)

    def test_density_2d(self):
        """
        Test that `density_2d()` returns the 2D Sersic function.
        :return:
        :rtype:
        """
        k_eff = 1
        R_sersic = 1
        n_sersic = 1
        e1 = 0.2
        e2 = 0.2
        center_x = 0.
        center_y = 0.

        n = 100
        x = np.logspace(-1, 1, n)
        y = np.logspace(-1, 1, n)

        X, Y = np.meshgrid(x, y)

        sersic_analytic = self.sersic_light.function(X, Y, k_eff, R_sersic,
                                                     n_sersic, e1, e2,
                                                     center_x, center_y)

        sersic_gauss = self.sersic_gauss.density_2d(X, Y, n_sersic, R_sersic,
                                                    k_eff, e1, e2, center_x,
                                                    center_y)

        assert np.all(
            np.abs(sersic_analytic - sersic_gauss) / np.sqrt(sersic_analytic)
            * 100 < 1.)

    def test_get_amps(self):
        """
        Test that `get_amps()` decomposes the Sersic profile within 1%
        Poission noise at R_sersic.
        :return:
        :rtype:
        """
        y = np.logspace(-1, 1, 100)

        k_eff = 1
        R_sersic = 1
        n_sersic = 1

        amps, sigmas = self.sersic_gauss.get_amps(n_sersic, R_sersic, k_eff)

        sersic = self.sersic_gauss.kappa_y(y, n_sersic, R_sersic, k_eff)

        back_sersic = np.zeros_like(y)

        for a, s in zip(amps, sigmas):
            back_sersic += a * np.exp(-y ** 2 / 2 / s ** 2)

        assert np.all(np.abs(sersic-back_sersic)/np.sqrt(sersic)*100 < 1.)


if __name__ == '__main__':
    pytest.main()
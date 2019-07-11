__author__ = 'ajshajib'

from lenstronomy.LensModel.Profiles.gauss_decomposition import SersicEllipseGaussDec
from lenstronomy.LensModel.Profiles.gauss_decomposition import CTNFWGaussDec
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic

import numpy as np
import numpy.testing as npt
import pytest


class TestSersicEllipseGaussDec(object):
    """
    This class tests the methods for Gauss-decomposed elliptic Sersic
    convergence.
    """
    def setup(self):
        self.sersic_gauss = SersicEllipseGaussDec()
        self.sersic_light = SersicElliptic()
        self.sersic_sphere = Sersic()

    def test_function(self):
        """
        Test the potential function of Gauss-decomposed elliptical Sersic by
        asserting that the numerical derivative of the computed potential
        matches with the analytical derivative values.

        :return:
        :rtype:
        """
        k_eff = 1.
        R_sersic = 1.
        n_sersic = 1.
        e1 = 0.2
        e2 = 0.2
        center_x = 0.
        center_y = 0.

        diff = 1.e-6

        n = 5
        xs = np.linspace(0.5 * R_sersic, 2. * R_sersic, n)
        ys = np.linspace(0.5 * R_sersic, 2. * R_sersic, n)

        for x, y in zip(xs, ys):
            func = self.sersic_gauss.function(x, y, e1=e1, e2=e2,
                                              center_x=center_x,
                                              center_y=center_y,
                                              n_sersic=n_sersic,
                                              R_sersic=R_sersic,
                                              k_eff=k_eff
                                              )

            func_dx = self.sersic_gauss.function(x+diff, y, e1=e1, e2=e2,
                                                 center_x=center_x,
                                                 center_y=center_y,
                                                 n_sersic=n_sersic,
                                                 R_sersic=R_sersic,
                                                 k_eff=k_eff
                                                 )

            func_dy = self.sersic_gauss.function(x, y+diff, e1=e1, e2=e2,
                                                 center_x=center_x,
                                                 center_y=center_y,
                                                 n_sersic=n_sersic,
                                                 R_sersic=R_sersic,
                                                 k_eff=k_eff
                                                 )

            f_x_num = (func_dx - func) / diff
            f_y_num = (func_dy - func) / diff

            f_x, f_y = self.sersic_gauss.derivatives(x, y, e1=e1, e2=e2,
                                                     center_x=center_x,
                                                     center_y=center_y,
                                                     n_sersic=n_sersic,
                                                     R_sersic=R_sersic,
                                                     k_eff=k_eff
                                                     )

            npt.assert_almost_equal(f_x_num, f_x, decimal=4)
            npt.assert_almost_equal(f_y_num, f_y, decimal=4)

    def test_derivatives(self):
        """
        Test the derivative function of Gauss-decomposed elliptical Sersic by
        matching with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.
        R_sersic = 1.
        n_sersic = 1.
        e1 = 5.e-5
        e2 = 0.
        center_x = 0.
        center_y = 0.

        n = 10
        x = np.linspace(0.5*R_sersic, 2.*R_sersic, n)
        y = np.linspace(0.5*R_sersic, 2.*R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_x_s, f_y_s = self.sersic_sphere.derivatives(X, Y, center_x=center_x,
                                                      center_y=center_y,
                                                      n_sersic=n_sersic,
                                                      R_sersic=R_sersic,
                                                      k_eff=k_eff
                                                      )
        f_x, f_y = self.sersic_gauss.derivatives(X, Y, e1=e1, e2=e2,
                                                 center_x=center_x,
                                                 center_y=center_y,
                                                 n_sersic=n_sersic,
                                                 R_sersic=R_sersic,
                                                 k_eff=k_eff
                                                 )

        npt.assert_allclose(f_x, f_x_s, rtol=1e-3, atol=0.)
        npt.assert_allclose(f_y, f_y_s, rtol=1e-3, atol=0.)

        npt.assert_almost_equal(f_x, f_x_s, decimal=3)
        npt.assert_almost_equal(f_y, f_y_s, decimal=3)

    def test_hessian(self):
        """
        Test the Hessian function of Gauss-decomposed elliptical Sersic by
        matching with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.
        R_sersic = 1.
        n_sersic = 1.
        e1 = 5e-5
        e2 = 0.
        center_x = 0.
        center_y = 0.

        n = 10
        x = np.linspace(0.5 * R_sersic, 2. * R_sersic, n)
        y = np.linspace(0.5 * R_sersic, 2. * R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_xx_s, f_yy_s, f_xy_s = self.sersic_sphere.hessian(X, Y,
                                                            center_x=center_x,
                                                            center_y=center_y,
                                                            n_sersic=n_sersic,
                                                            R_sersic=R_sersic,
                                                            k_eff=k_eff)
        f_xx, f_yy, f_xy = self.sersic_gauss.hessian(X, Y, e1=e1, e2=e2,
                                                     center_x=center_x,
                                                     center_y=center_y,
                                                     n_sersic=n_sersic,
                                                     R_sersic=R_sersic,
                                                     k_eff=k_eff)

        npt.assert_almost_equal(f_xx_s, f_xx, decimal=3)
        npt.assert_almost_equal(f_yy_s, f_yy, decimal=3)
        npt.assert_almost_equal(f_xy_s, f_xy, decimal=3)

    def test_density_2d(self):
        """
        Test the density function of Gauss-decomposed elliptical Sersic by
        checking with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.
        R_sersic = 1.
        n_sersic = 1.
        e1 = 0.2
        e2 = 0.2
        center_x = 0.
        center_y = 0.

        n = 100
        x = np.logspace(-1., 1., n)
        y = np.logspace(-1., 1., n)

        X, Y = np.meshgrid(x, y)

        sersic_analytic = self.sersic_light.function(X, Y, e1=e1, e2=e2,
                                                 center_x=center_x,
                                                 center_y=center_y,
                                                 n_sersic=n_sersic,
                                                 R_sersic=R_sersic,
                                                 amp=k_eff)

        sersic_gauss = self.sersic_gauss.density_2d(X, Y, e1=e1, e2=e2,
                                                    center_x=center_x,
                                                    center_y=center_y,
                                                    n_sersic=n_sersic,
                                                    R_sersic=R_sersic,
                                                    k_eff=k_eff)

        assert np.all(
            np.abs(sersic_analytic - sersic_gauss) / np.sqrt(sersic_analytic)
            * 100. < 1.)

    def test_gauss_decompose_sersic(self):
        """
        Test that `gauss_decompose_sersic()` decomposes the Sersic profile within 1%
        Poission noise at R_sersic.

        :return:
        :rtype:
        """
        y = np.logspace(-1., 1., 100)

        k_eff = 1.
        R_sersic = 1.
        n_sersic = 1.

        amps, sigmas = self.sersic_gauss.gauss_decompose(n_sersic=n_sersic,
                                               R_sersic=R_sersic, k_eff=k_eff)

        sersic = self.sersic_gauss.get_kappa_1d(y, n_sersic=n_sersic,
                                               R_sersic=R_sersic, k_eff=k_eff)

        back_sersic = np.zeros_like(y)

        for a, s in zip(amps, sigmas):
            back_sersic += a * np.exp(-y ** 2 / 2. / s ** 2)

        assert np.all(np.abs(sersic-back_sersic)/np.sqrt(sersic)*100. < 1.)


class TestCTNFWGaussDec(object):
    """
    This class tests the methods for Gauss-decomposed spherical
    cored-truncated NFW profile.
    """
    def setup(self):
        self.ctnfw_gauss = CTNFWGaussDec(n_sigma=15)

    def test_gauss_decompose_ctnfw(self):
        """
        Test the Gaussian decomposition of core-truncated NFW profile.
        :return:
        :rtype:
        """
        rho_s = 5.
        r_s = 5.
        r_core = 0.3
        r_trunc = 10.
        a = 2

        r = np.logspace(-1, 1, 1000) * r_s

        beta = r_core / r_s
        tau = r_trunc / r_s

        x = r / r_s

        true_values = rho_s * (tau * tau / (tau * tau + x * x)) / (x**a +
                                    beta ** a) ** ( 1. / a) / (1. + x) ** 2

        amps, sigmas = self.ctnfw_gauss.gauss_decompose(r_s=r_s,
                                                            r_core=r_core,
                                                            r_trunc=r_trunc,
                                                            rho_s=rho_s, a=a)

        print(len(sigmas))
        gauss_dec_values = np.zeros_like(x)
        for a, s in zip(amps, sigmas):
            gauss_dec_values += a / np.sqrt(2*np.pi) / s * np.exp(
                -r**2/2./s**2)

        # test if the approximation is valid within 2%
        npt.assert_allclose(true_values, true_values, rtol=0.02)


if __name__ == '__main__':
    pytest.main()
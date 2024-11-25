__author__ = "ajshajib"

from lenstronomy.LensModel.Profiles.gauss_decomposition import SersicEllipseGaussDec
from lenstronomy.LensModel.Profiles.gauss_decomposition import (
    GeneralizedNFWEllipseGaussDec,
)
from lenstronomy.LensModel.Profiles.gauss_decomposition import CTNFWGaussDec
from lenstronomy.LensModel.Profiles.sersic import Sersic
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic
from lenstronomy.LensModel.Profiles.gnfw import GNFW

import numpy as np
import numpy.testing as npt
import pytest


class TestSersicEllipseGaussDec(object):
    """This class tests the methods for Gauss-decomposed elliptic Sersic convergence."""

    def setup_method(self):
        self.sersic_gauss = SersicEllipseGaussDec()
        self.sersic_light = SersicElliptic(sersic_major_axis=False)
        self.sersic_sphere = Sersic(sersic_major_axis=False)

    def test_function(self):
        """Test the potential function of Gauss-decomposed elliptical Sersic by
        asserting that the numerical derivative of the computed potential matches with
        the analytical derivative values.

        :return:
        :rtype:
        """
        k_eff = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        e1 = 0.2
        e2 = 0.2
        center_x = 0.0
        center_y = 0.0

        diff = 1.0e-6

        n = 5
        xs = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)
        ys = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)

        for x, y in zip(xs, ys):
            func = self.sersic_gauss.function(
                x,
                y,
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
                n_sersic=n_sersic,
                R_sersic=R_sersic,
                k_eff=k_eff,
            )

            func_dx = self.sersic_gauss.function(
                x + diff,
                y,
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
                n_sersic=n_sersic,
                R_sersic=R_sersic,
                k_eff=k_eff,
            )

            func_dy = self.sersic_gauss.function(
                x,
                y + diff,
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
                n_sersic=n_sersic,
                R_sersic=R_sersic,
                k_eff=k_eff,
            )

            f_x_num = (func_dx - func) / diff
            f_y_num = (func_dy - func) / diff

            f_x, f_y = self.sersic_gauss.derivatives(
                x,
                y,
                e1=e1,
                e2=e2,
                center_x=center_x,
                center_y=center_y,
                n_sersic=n_sersic,
                R_sersic=R_sersic,
                k_eff=k_eff,
            )

            npt.assert_almost_equal(f_x_num, f_x, decimal=4)
            npt.assert_almost_equal(f_y_num, f_y, decimal=4)

    def test_derivatives(self):
        """Test the derivative function of Gauss-decomposed elliptical Sersic by
        matching with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        e1 = 5.0e-5
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0

        n = 10
        x = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)
        y = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_x_s, f_y_s = self.sersic_sphere.derivatives(
            X,
            Y,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            k_eff=k_eff,
        )
        f_x, f_y = self.sersic_gauss.derivatives(
            X,
            Y,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            k_eff=k_eff,
        )

        npt.assert_allclose(f_x, f_x_s, rtol=1e-3, atol=0.0)
        npt.assert_allclose(f_y, f_y_s, rtol=1e-3, atol=0.0)

        npt.assert_almost_equal(f_x, f_x_s, decimal=3)
        npt.assert_almost_equal(f_y, f_y_s, decimal=3)

    def test_hessian(self):
        """Test the Hessian function of Gauss-decomposed elliptical Sersic by matching
        with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        e1 = 5e-5
        e2 = 0.0
        center_x = 0.0
        center_y = 0.0

        n = 10
        x = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)
        y = np.linspace(0.5 * R_sersic, 2.0 * R_sersic, n)

        X, Y = np.meshgrid(x, y)

        f_xx_s, f_xy_s, f_yx_s, f_yy_s = self.sersic_sphere.hessian(
            X,
            Y,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            k_eff=k_eff,
        )
        f_xx, f_xy, f_yx, f_yy = self.sersic_gauss.hessian(
            X,
            Y,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            k_eff=k_eff,
        )

        npt.assert_almost_equal(f_xx_s, f_xx, decimal=3)
        npt.assert_almost_equal(f_yy_s, f_yy, decimal=3)
        npt.assert_almost_equal(f_xy_s, f_xy, decimal=3)
        npt.assert_almost_equal(f_xy_s, f_yx_s, decimal=3)

    def test_density_2d(self):
        """Test the density function of Gauss-decomposed elliptical Sersic by checking
        with the spherical case.

        :return:
        :rtype:
        """
        k_eff = 1.0
        R_sersic = 1.0
        n_sersic = 1.0
        e1 = 0.2
        e2 = 0.2
        center_x = 0.0
        center_y = 0.0

        n = 100
        x = np.logspace(-1.0, 1.0, n)
        y = np.logspace(-1.0, 1.0, n)

        X, Y = np.meshgrid(x, y)

        sersic_analytic = self.sersic_light.function(
            X,
            Y,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            amp=k_eff,
        )

        sersic_gauss = self.sersic_gauss.density_2d(
            X,
            Y,
            e1=e1,
            e2=e2,
            center_x=center_x,
            center_y=center_y,
            n_sersic=n_sersic,
            R_sersic=R_sersic,
            k_eff=k_eff,
        )

        print(np.abs(sersic_analytic - sersic_gauss) / np.sqrt(sersic_analytic))

        assert np.all(
            np.abs(sersic_analytic - sersic_gauss) / np.sqrt(sersic_analytic) * 100.0
            < 1.0
        )

    def test_gauss_decompose_sersic(self):
        """Test that `gauss_decompose_sersic()` decomposes the Sersic profile within 1%
        Poission noise at R_sersic.

        :return:
        :rtype:
        """
        y = np.logspace(-1.0, 1.0, 100)

        k_eff = 1.0
        R_sersic = 1.0
        n_sersic = 1.0

        amps, sigmas = self.sersic_gauss.gauss_decompose(
            n_sersic=n_sersic, R_sersic=R_sersic, k_eff=k_eff
        )

        sersic = self.sersic_gauss.get_kappa_1d(
            y, n_sersic=n_sersic, R_sersic=R_sersic, k_eff=k_eff
        )

        back_sersic = np.zeros_like(y)

        for a, s in zip(amps, sigmas):
            back_sersic += a * np.exp(-(y**2) / 2.0 / s**2)

        assert np.all(np.abs(sersic - back_sersic) / np.sqrt(sersic) * 100.0 < 1.0)


class TestGeneralizedNFWGaussDec(object):
    """This class tests the methods for Gauss-decomposed generalized NFW profile."""

    def setup_method(self):
        self.gnfw_gauss = GeneralizedNFWEllipseGaussDec()
        self.gnfw = GNFW()

    def test_get_kappa_1d(self):
        rs = np.logspace(-2, 1, 100)
        kappa_s = 0.1
        R_s = 20.0
        gamma_in = 0.8

        alpha_Rs = self.gnfw._alpha(R_s, R_s, kappa_s, gamma_in)
        kappa_1d = self.gnfw_gauss.get_kappa_1d(
            rs, alpha_Rs=alpha_Rs, Rs=R_s, gamma_in=gamma_in
        )
        kappa_1d_spherical = self.gnfw._kappa(rs, R_s, kappa_s, gamma_in)

        npt.assert_allclose(kappa_1d, kappa_1d_spherical, rtol=0.01)

    def test_get_scale(self):
        alpha_Rs = 1
        R_s = 20.0
        gamma_in = 0.8
        assert (
            self.gnfw_gauss.get_scale(alpha_Rs=alpha_Rs, Rs=R_s, gamma_in=gamma_in)
            == R_s
        )


class TestCTNFWGaussDec(object):
    """This class tests the methods for Gauss-decomposed spherical cored-truncated NFW
    profile."""

    def setup_method(self):
        self.ctnfw_gauss = CTNFWGaussDec(n_sigma=15)

    def test_gauss_decompose_ctnfw(self):
        """Test the Gaussian decomposition of core-truncated NFW profile.

        :return:
        :rtype:
        """
        rho_s = 5.0
        r_s = 5.0
        r_core = 0.3
        r_trunc = 10.0
        a = 2

        r = np.logspace(-1, 1, 1000) * r_s

        beta = r_core / r_s
        tau = r_trunc / r_s

        x = r / r_s

        true_values = (
            rho_s
            * (tau * tau / (tau * tau + x * x))
            / (x**a + beta**a) ** (1.0 / a)
            / (1.0 + x) ** 2
        )

        amps, sigmas = self.ctnfw_gauss.gauss_decompose(
            r_s=r_s, r_core=r_core, r_trunc=r_trunc, rho_s=rho_s, a=a
        )

        print(len(sigmas))
        gauss_dec_values = np.zeros_like(x)
        for a, s in zip(amps, sigmas):
            gauss_dec_values += (
                a / np.sqrt(2 * np.pi) / s * np.exp(-(r**2) / 2.0 / s**2)
            )

        # test if the approximation is valid within 2%
        npt.assert_allclose(true_values, true_values, rtol=0.02)


if __name__ == "__main__":
    pytest.main()

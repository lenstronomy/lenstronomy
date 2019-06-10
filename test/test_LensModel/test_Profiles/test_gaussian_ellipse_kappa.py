__author__ = 'ajshajib'

from lenstronomy.LensModel.Profiles.gaussian_ellipse_kappa import GaussianEllipseKappa
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa

import numpy as np
import numpy.testing as npt
from scipy.special import expi
from scipy.special import wofz
import pytest


class TestGaussianEllipseKappa(object):
    """
    This class tests the methods for elliptical Gaussian convergence.
    """
    def setup(self):
        """
        :return:
        :rtype:
        """
        self.gaussian_kappa = GaussianKappa()
        self.gaussian_kappa_ellipse = GaussianEllipseKappa()

    def test_function(self):
        """
        Test the `function()` method at the spherical limit.

        :return:
        :rtype:
        """
        # almost spherical case
        x = 1.
        y = 1.
        e1, e2 = 5e-5, 0.
        sigma = 1.
        amp = 2.

        f_ = self.gaussian_kappa_ellipse.function(x, y, amp, sigma, e1, e2)

        r2 = x*x + y*y
        f_sphere = amp/(2.*np.pi*sigma**2) * sigma**2 * (np.euler_gamma -
                        expi(-r2/2./sigma**2) + np.log(r2/2./sigma**2))

        npt.assert_almost_equal(f_, f_sphere, decimal=4)

        # spherical case
        e1, e2 = 0., 0.
        f_ = self.gaussian_kappa_ellipse.function(x, y, amp, sigma, e1, e2)

        npt.assert_almost_equal(f_, f_sphere, decimal=4)



    def test_derivatives(self):
        """
        Test the `derivatives()` method at the spherical limit.

        :return:
        :rtype:
        """
        # almost spherical case
        x = 1.
        y = 1.
        e1, e2 = 5e-5, 0.
        sigma = 1.
        amp = 2.

        f_x, f_y = self.gaussian_kappa_ellipse.derivatives(x, y, amp, sigma,
                                                           e1, e2)
        f_x_sphere, f_y_sphere = self.gaussian_kappa.derivatives(x, y, amp=amp,
                                                                 sigma=sigma)
        npt.assert_almost_equal(f_x, f_x_sphere, decimal=4)
        npt.assert_almost_equal(f_y, f_y_sphere, decimal=4)

        # spherical case
        e1, e2 = 0., 0.
        f_x, f_y = self.gaussian_kappa_ellipse.derivatives(x, y, amp, sigma,
                                                           e1, e2)

        npt.assert_almost_equal(f_x, f_x_sphere, decimal=4)
        npt.assert_almost_equal(f_y, f_y_sphere, decimal=4)

    def test_hessian(self):
        """
        Test the `hessian()` method at the spherical limit.

        :return:
        :rtype:
        """
        # almost spherical case
        x = 1.
        y = 1.
        e1, e2 = 5e-5, 0.
        sigma = 1.
        amp = 2.

        f_xx, f_yy, f_xy = self.gaussian_kappa_ellipse.hessian(x, y, amp,
                                                               sigma, e1, e2)
        f_xx_sphere, f_yy_sphere, f_xy_sphere = self.gaussian_kappa.hessian(x,
                                                       y, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_xx, f_xx_sphere, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_sphere, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_sphere, decimal=4)

        # spherical case
        e1, e2 = 0., 0.
        f_xx, f_yy, f_xy = self.gaussian_kappa_ellipse.hessian(x, y, amp,
                                                               sigma, e1, e2)

        npt.assert_almost_equal(f_xx, f_xx_sphere, decimal=4)
        npt.assert_almost_equal(f_yy, f_yy_sphere, decimal=4)
        npt.assert_almost_equal(f_xy, f_xy_sphere, decimal=4)

    def test_density_2d(self):
        """
        Test the `density_2d()` method at the spherical limit.

        :return:
        :rtype:
        """
        # almost spherical case
        x = 1.
        y = 1.
        e1, e2 = 5e-5, 0.
        sigma = 1.
        amp = 2.
        f_ = self.gaussian_kappa_ellipse.density_2d(x, y, amp, sigma, e1, e2)
        f_sphere = amp / (2.*np.pi*sigma**2) * np.exp(-(x*x+y*y)/2./sigma**2)
        npt.assert_almost_equal(f_, f_sphere, decimal=4)

    def test_w_f_approx(self):
        """
        Test the `w_f_approx()` method with values computed using
        `scipy.special.wofz()`.

        :return:
        :rtype:
        """
        x = np.logspace(-3., 3., 100)
        y = np.logspace(-3., 3., 100)

        X, Y = np.meshgrid(x, y)

        w_f_app = self.gaussian_kappa_ellipse.w_f_approx(X+1j*Y)
        w_f_scipy = wofz(X+1j*Y)

        npt.assert_allclose(w_f_app.real, w_f_scipy.real, rtol=4e-5, atol=0)
        npt.assert_allclose(w_f_app.imag, w_f_scipy.imag, rtol=4e-5, atol=0)

        # check `derivatives()` method with and without `scipy.special.wofz()`
        x = 1.
        y = 1.
        e1, e2 = 5e-5, 0
        sigma = 1.
        amp = 2.

        # with `scipy.special.wofz()`
        gauss_scipy = GaussianEllipseKappa(use_scipy_wofz=True)
        f_x_sp, f_y_sp = gauss_scipy.derivatives(x, y, amp, sigma, e1, e2)

        # with `GaussEllipseKappa.w_f_approx()`
        gauss_approx = GaussianEllipseKappa(use_scipy_wofz=False)
        f_x_ap, f_y_ap = gauss_approx.derivatives(x, y, amp, sigma, e1, e2)

        npt.assert_almost_equal(f_x_sp, f_x_ap, decimal=4)
        npt.assert_almost_equal(f_y_sp, f_y_ap, decimal=4)


if __name__ == '__main__':
    pytest.main()
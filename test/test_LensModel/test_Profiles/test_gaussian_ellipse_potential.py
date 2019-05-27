__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.gaussian_ellipse_potential import GaussianEllipsePotential
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa

import numpy as np
import numpy.testing as npt
import pytest


class TestGaussianKappaPot(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.gaussian_kappa = GaussianKappa()
        self.ellipse = GaussianEllipsePotential()

    def test_function(self):
        x = 1
        y = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_ = self.ellipse.function(x, y, amp, sigma, e1, e2)
        f_sphere = self.gaussian_kappa.function(x, y, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_, f_sphere, decimal=8)

    def test_derivatives(self):
        x = 1
        y = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_x, f_y = self.ellipse.derivatives(x, y, amp, sigma, e1, e2)
        f_x_sphere, f_y_sphere = self.gaussian_kappa.derivatives(x, y, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_x, f_x_sphere, decimal=8)
        npt.assert_almost_equal(f_y, f_y_sphere, decimal=8)

    def test_hessian(self):
        x = 1
        y = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_xx, f_yy, f_xy = self.ellipse.hessian(x, y, amp, sigma, e1, e2)
        f_xx_sphere, f_yy_sphere, f_xy_sphere = self.gaussian_kappa.hessian(x, y, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_xx, f_xx_sphere, decimal=5)
        npt.assert_almost_equal(f_yy, f_yy_sphere, decimal=5)
        npt.assert_almost_equal(f_xy, f_xy_sphere, decimal=5)

    def test_density_2d(self):
        x = 1
        y = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_ = self.ellipse.density_2d(x, y, amp, sigma, e1, e2)
        f_sphere = self.gaussian_kappa.density_2d(x, y, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_, f_sphere, decimal=8)

    def test_mass_2d(self):
        r = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_ = self.ellipse.mass_2d(r, amp, sigma, e1, e2)
        f_sphere = self.gaussian_kappa.mass_2d(r, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_, f_sphere, decimal=8)

    def test_mass_2d_lens(self):
        r = 1
        e1, e2 = 0, 0
        sigma = 1
        amp = 1
        f_ = self.ellipse.mass_2d_lens(r, amp, sigma, e1, e2)
        f_sphere = self.gaussian_kappa.mass_2d_lens(r, amp=amp, sigma=sigma)
        npt.assert_almost_equal(f_, f_sphere, decimal=8)


if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.gaussian_potential import Gaussian
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa

import numpy as np
import numpy.testing as npt
import pytest

class TestGaussian(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.Gaussian = Gaussian()

    def test_function(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.function(x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values == np.exp(-1./2)
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.function(x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == np.exp(-1./2)
        assert values[1] == np.exp(-2.**2/2)
        assert values[2] == np.exp(-3.**2/2)

    def test_derivatives(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.derivatives( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == 0.
        assert values[1] == -np.exp(-1./2)
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.derivatives( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0][0] == -np.exp(-1./2)
        assert values[1][0] == 0.
        assert values[0][1] == -2*np.exp(-2.**2/2)
        assert values[1][1] == 0.

    def test_hessian(self):
        x = 1
        y = 2
        amp = 1.*2*np.pi
        center_x = 1.
        center_y = 1.
        sigma_x = 1.
        sigma_y = 1.
        values = self.Gaussian.hessian( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0] == -np.exp(-1./2)
        assert values[1] == 0.
        assert values[2] == 0.
        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.Gaussian.hessian( x, y, amp, center_x, center_y, sigma_x, sigma_y)
        assert values[0][0] == 0.
        assert values[1][0] == -np.exp(-1./2)
        assert values[2][0] == 0.
        assert values[0][1] == 0.40600584970983811
        assert values[1][1] == -0.1353352832366127
        assert values[2][1] == 0.


class TestGaussianKappa(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.gaussian_kappa = GaussianKappa()
        self.gaussian = Gaussian()

    def test_derivatives(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = 1.*2*np.pi
        center_x = 0.
        center_y = 0.
        sigma = 1.
        f_x, f_y = self.gaussian_kappa.derivatives(x, y, amp, sigma, center_x, center_y)
        npt.assert_almost_equal(f_x[2], 0.63813558702212059, decimal=8)
        npt.assert_almost_equal(f_y[2], 0.63813558702212059, decimal=8)

    def test_hessian(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = 1.*2*np.pi
        center_x = 0.
        center_y = 0.
        sigma = 1.

        f_xx, f_yy, f_xy = self.gaussian_kappa.hessian(x, y, amp, sigma, center_x, center_y)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_true = self.gaussian.function(x, y, amp, sigma, sigma, center_x, center_y)
        print(kappa_true)
        print(kappa)
        npt.assert_almost_equal(kappa[0], kappa_true[0], decimal=5)
        npt.assert_almost_equal(kappa[1], kappa_true[1], decimal=5)

    def test_density_2d(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = 1.*2*np.pi
        center_x = 0.
        center_y = 0.
        sigma = 1.
        f_xx, f_yy, f_xy = self.gaussian_kappa.hessian(x, y, amp, sigma, center_x, center_y)
        kappa = 1./2 * (f_xx + f_yy)
        amp_3d = self.gaussian_kappa._amp2d_to_3d(amp, sigma, sigma)
        density_2d = self.gaussian_kappa.density_2d(x, y, amp_3d, sigma, center_x, center_y)
        npt.assert_almost_equal(kappa[1], density_2d[1], decimal=5)
        npt.assert_almost_equal(kappa[2], density_2d[2], decimal=5)

    def test_3d_2d_convention(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = 1.*2*np.pi
        center_x = 0.
        center_y = 0.
        sigma = 1.
        amp_3d = self.gaussian_kappa._amp2d_to_3d(amp, sigma, sigma)
        density_2d_gauss = self.gaussian_kappa.density_2d(x, y, amp_3d, sigma, center_x, center_y)
        density_2d = self.gaussian.function(x, y, amp, sigma, sigma, center_x, center_y)
        npt.assert_almost_equal(density_2d_gauss[1], density_2d[1], decimal=5)


if __name__ == '__main__':
    pytest.main()

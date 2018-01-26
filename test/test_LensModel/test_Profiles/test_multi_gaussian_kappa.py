__author__ = 'sibirrer'


from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussian_kappa

import numpy as np
import numpy.testing as npt
import pytest


class TestGaussianKappa(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.gaussian_kappa = MultiGaussian_kappa()
        self.gaussian = Gaussian()
        self.g_kappa = GaussianKappa()

    def test_derivatives(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = [1.*2*np.pi]
        center_x = 0.
        center_y = 0.
        sigma = [1.]
        f_x, f_y = self.gaussian_kappa.derivatives(x, y, amp, sigma, center_x, center_y)
        npt.assert_almost_equal(f_x[2], 0.63813558702212059, decimal=8)
        npt.assert_almost_equal(f_y[2], 0.63813558702212059, decimal=8)

    def test_hessian(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = [1.*2*np.pi]
        center_x = 0.
        center_y = 0.
        sigma = [1.]
        f_xx, f_yy, f_xy = self.gaussian_kappa.hessian(x, y, amp, sigma, center_x, center_y)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_true = self.gaussian.function(x, y, amp[0], sigma[0], sigma[0], center_x, center_y)
        print(kappa_true)
        print(kappa)
        npt.assert_almost_equal(kappa[0], kappa_true[0], decimal=5)
        npt.assert_almost_equal(kappa[1], kappa_true[1], decimal=5)

    def test_density_2d(self):
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        amp = [1.*2*np.pi]
        center_x = 0.
        center_y = 0.
        sigma = [1.]
        f_xx, f_yy, f_xy = self.gaussian_kappa.hessian(x, y, amp, sigma, center_x, center_y)
        kappa = 1./2 * (f_xx + f_yy)
        amp_3d = self.g_kappa._amp2d_to_3d(amp, sigma[0], sigma[0])
        density_2d = self.gaussian_kappa.density_2d(x, y, amp_3d, sigma, center_x, center_y)
        npt.assert_almost_equal(kappa[1], density_2d[1], decimal=5)
        npt.assert_almost_equal(kappa[2], density_2d[2], decimal=5)

    def test_density(self):
        amp = [1.*2*np.pi]

        sigma = [1.]
        density = self.gaussian_kappa.density(1., amp, sigma)
        npt.assert_almost_equal(density, 0.6065306597126334, decimal=8)


if __name__ == '__main__':
    pytest.main()
__author__ = 'sibirrer'


from lenstronomy.LightModel.Profiles.gaussian import Gaussian
from lenstronomy.LensModel.Profiles.gaussian_kappa import GaussianKappa
from lenstronomy.LensModel.Profiles.gaussian_kappa_ellipse import GaussianKappaEllipse
from lenstronomy.LensModel.Profiles.multi_gaussian_kappa import MultiGaussianKappa, MultiGaussianKappaEllipse

import numpy as np
import numpy.testing as npt
import pytest


class TestGaussianKappa(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.gaussian_kappa = MultiGaussianKappa()
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


class TestGaussianKappaEllipse(object):
    """
    test the Gaussian with Gaussian kappa
    """
    def setup(self):
        self.multi = MultiGaussianKappaEllipse()
        self.single = GaussianKappaEllipse()

    def test_function(self):
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_ = self.multi.function(x, y, amp=[amp], sigma=[sigma], e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        f_single = self.single.function(x, y, amp=amp, sigma=sigma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        npt.assert_almost_equal(f_, f_single, decimal=8)

    def test_derivatives(self):
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_x, f_y = self.multi.derivatives(x, y, amp=[amp], sigma=[sigma], e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        f_x_s, f_y_s = self.single.derivatives(x, y, amp=amp, sigma=sigma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        npt.assert_almost_equal(f_x, f_x_s, decimal=8)
        npt.assert_almost_equal(f_y, f_y_s, decimal=8)

    def test_hessian(self):
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_xx, f_yy, f_xy = self.multi.hessian(x, y, amp=[amp], sigma=[sigma], e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        f_xx_s, f_yy_s, f_xy_s = self.single.hessian(x, y, amp=amp, sigma=sigma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        npt.assert_almost_equal(f_xx, f_xx_s, decimal=8)
        npt.assert_almost_equal(f_yy, f_yy_s, decimal=8)
        npt.assert_almost_equal(f_xy, f_xy_s, decimal=8)

    def test_density_2d(self):
        x, y = 1, 2
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        center_x, center_y = 1, 0
        f_ = self.multi.density_2d(x, y, amp=[amp], sigma=[sigma], e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        f_single = self.single.density_2d(x, y, amp=amp, sigma=sigma, e1=e1, e2=e2, center_x=center_x, center_y=center_y)
        npt.assert_almost_equal(f_, f_single, decimal=8)

    def test_density(self):
        r = 1
        amp = 1
        sigma = 1
        e1, e2 = 0.1, -0.1
        f_ = self.multi.density(r, amp=[amp], sigma=[sigma], e1=e1, e2=e2)
        f_single = self.single.density(r, amp=amp, sigma=sigma, e1=e1, e2=e2)
        npt.assert_almost_equal(f_, f_single, decimal=8)


if __name__ == '__main__':
    pytest.main()
__author__ = 'lucateo'

from lenstronomy.LensModel.Profiles.cored_density_exp import CoredDensityExp

import numpy as np
import numpy.testing as npt
import pytest


class TestCoredDensityExp(object):
    """Tests the Gaussian methods."""
    def setup_method(self):
        self.model = CoredDensityExp()

    def test_function(self):
        r = np.linspace(start=0.01, stop=2, num=10)
        kappa_0 = 0.1
        theta_c = 5
        f_ = self.model.function(r, 0, kappa_0, theta_c)
        delta = 0.0001
        f_d = self.model.function(r + delta, 0, kappa_0, theta_c)
        f_x_num = (f_d - f_) / delta
        f_x, _ = self.model.derivatives(r, 0, kappa_0, theta_c)
        npt.assert_almost_equal(f_x_num, f_x, decimal=3)
        # Try MSD limit
        theta_c_large = 13
        f_reference = self.model.function(0, 0, kappa_0, theta_c_large)
        f_large = self.model.function(r, 0, kappa_0, theta_c_large)
        f_MSD = 0.5* kappa_0 * r**2
        npt.assert_almost_equal(f_large - f_reference, f_MSD, decimal=3)

    def test_derivatives(self):
        x = 0.5
        y = 0.8
        r = np.sqrt(x**2 + y**2)
        kappa_0, theta_c = 0.2, 9 # Trying MSD limit
        f_x, f_y = self.model.derivatives( x, y, kappa_0, theta_c)
        alpha_MSD = kappa_0 * r
        npt.assert_almost_equal(f_x, alpha_MSD * x/r, decimal=3)
        npt.assert_almost_equal(f_y, alpha_MSD * y/r, decimal=3)

    def test_hessian(self):
        x = np.linspace(start=0.01, stop=100, num=100)
        y = 0
        r = np.sqrt(x**2 + y**2)
        kappa_0 = 0.12
        theta_c = 6
        f_xx, f_xy, f_yx, f_yy = self.model.hessian(x, y, kappa_0, theta_c)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_direct = self.model.kappa_r(r, kappa_0, theta_c)
        npt.assert_almost_equal(kappa, kappa_direct, decimal=5)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

    def test_mass_3d(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        r = np.sqrt(x ** 2 + y ** 2)
        kappa_0 = 0.1
        theta_c = 7
        m3d = self.model.mass_3d(r, kappa_0, theta_c)
        m3d_lens = self.model.mass_3d_lens(r, kappa_0, theta_c)
        npt.assert_almost_equal(m3d, m3d_lens, decimal=8)


if __name__ == '__main__':
    pytest.main()

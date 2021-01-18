__author__ = 'lucateo'

from lenstronomy.LensModel.Profiles.uldm import Uldm

import numpy as np
import numpy.testing as npt
import pytest


class TestUldm(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.model = Uldm()

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

    def test_derivatives(self):
        pass

    def test_hessian(self):
        x = np.linspace(start=0.01, stop=100, num=100)
        y = 0
        r = np.sqrt(x**2 + y**2)
        kappa_0 = 0.12
        theta_c = 6
        f_xx, f_yy, f_xy = self.model.hessian(x, y, kappa_0, theta_c)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_direct = self.model.kappa_r(r, kappa_0, theta_c)
        npt.assert_almost_equal(kappa, kappa_direct, decimal=5)

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


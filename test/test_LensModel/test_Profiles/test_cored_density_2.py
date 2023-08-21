__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.cored_density_2 import CoredDensity2

import numpy as np
import numpy.testing as npt
import pytest


class TestCoredDensity(object):
    """Tests the Gaussian methods."""
    def setup_method(self):
        self.model = CoredDensity2()

    def test_function(self):
        r = np.linspace(start=0.01, stop=10, num=10)
        sigma0 = 0.2
        r_core = 5
        f_ = self.model.function(r, 0, sigma0, r_core)
        delta = 0.00001
        f_d = self.model.function(r + delta, 0, sigma0, r_core)
        f_x_num = (f_d - f_) / delta
        f_x, _ = self.model.derivatives(r, 0, sigma0, r_core)
        npt.assert_almost_equal(f_x_num, f_x, decimal=3)

        #test single value vs list of outputs
        f_ = self.model.function(1, 0, sigma0, r_core)
        f_list = self.model.function(np.array([1]), 0, sigma0, r_core)
        npt.assert_almost_equal(f_, f_list[0], decimal=8)

    def test_derivatives(self):
        pass

    def test_dalpha_dr(self):
        x = np.array([1., 3., 4.])
        y = np.array([2., 1., 1.])
        r = np.sqrt(x ** 2 + y ** 2)
        sigma0 = 0.1
        r_core = 7.
        dalpha_dr = self.model.d_alpha_dr(r, sigma0, r_core)
        alpha_r = self.model.alpha_r(r, sigma0, r_core)
        delta = 0.00001
        d_alpha_r = self.model.alpha_r(r + delta, sigma0, r_core)
        d_alpha_dr_num = (d_alpha_r - alpha_r) / delta
        npt.assert_almost_equal(dalpha_dr, d_alpha_dr_num)

    def test_hessian(self):

        x = np.linspace(start=0.1, stop=10, num=100)
        y = 0
        r = np.sqrt(x**2 + y**2)
        sigma0 = 0.1
        r_core = 2.
        f_xx, f_xy, f_yx, f_yy = self.model.hessian(x, y, sigma0, r_core)
        kappa = 1./2 * (f_xx + f_yy)
        kappa_direct = self.model.kappa_r(r, sigma0, r_core)
        npt.assert_almost_equal(kappa/kappa_direct, 1, decimal=5)
        npt.assert_almost_equal(f_xy, f_yx, decimal=8)

    def test_mass_3d(self):
        x = np.array([1, 3, 4])
        y = np.array([2, 1, 1])
        r = np.sqrt(x ** 2 + y ** 2)
        sigma0 = 0.1
        r_core = 7
        m3d = self.model.mass_3d(r, sigma0, r_core)
        m3d_lens = self.model.mass_3d_lens(r, sigma0, r_core)
        npt.assert_almost_equal(m3d, m3d_lens, decimal=8)


if __name__ == '__main__':
    pytest.main()

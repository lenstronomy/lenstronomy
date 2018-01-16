__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.sersic_double import SersicDouble

import numpy as np
import pytest
import numpy.testing as npt


class TestSersic(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.sersic = SersicDouble()

    def test_function(self):
        x = 1
        y = 2
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        flux_ratio = 2
        phi_G = 0
        q = 0.9
        R_2 = 0.1
        n_2 = 1.
        phi_G_2 = 1
        q_2 = 1
        values = self.sersic.function(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        npt.assert_almost_equal(values, 1.1002728151856438, decimal=10)

        x = np.array([0])
        y = np.array([0])
        values = self.sersic.function(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 0., decimal=10)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.sersic.function(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        npt.assert_almost_equal(values[0], 1.0505646128277208, decimal=10)
        npt.assert_almost_equal(values[1], 1.3475405549572042, decimal=10)
        npt.assert_almost_equal(values[2], 1.5974687674142451, decimal=10)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        flux_ratio = 2
        phi_G = 0
        q = 0.9
        R_2 = 0.1
        n_2 = 1.
        phi_G_2 = 1
        q_2 = 1
        f_x, f_y = self.sersic.derivatives(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        assert f_x[0] == 0.14520470528563453
        assert f_y[0] == 0.35359090338854549
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.sersic.derivatives(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.derivatives(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        assert values[0][0] == 0.14520470528563453
        assert values[1][0] == 0.35359090338854549
        assert values[0][1] == 0.27295610390132041
        assert values[1][1] == 0.11086585568793469

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        n_sersic = 2.
        r_eff = 1.
        k_eff = 0.2
        flux_ratio = 2
        phi_G = 0
        q = 0.9
        R_2 = 0.1
        n_2 = 1.
        phi_G_2 = 1
        q_2 = 1
        f_xx, f_yy,f_xy = self.sersic.hessian(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        assert f_xx[0] == 0.10485730465512044
        npt.assert_almost_equal(f_yy[0], -0.061885450483752102, decimal=10)
        npt.assert_almost_equal(f_xy[0], -0.098085243120313303, decimal=10)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.sersic.hessian(x, y, k_eff, flux_ratio, r_eff, n_sersic, phi_G, q, R_2, n_2, phi_G_2, q_2, center_x=0, center_y=0)
        assert values[0][0] == 0.10485730465512044
        npt.assert_almost_equal(values[1][0], -0.061885450483752102, decimal=10)
        npt.assert_almost_equal(values[2][0], -0.098085243120313303, decimal=10)
        npt.assert_almost_equal(values[0][1], -0.048462684289207747, decimal=10)
        npt.assert_almost_equal(values[1][1], 0.087870563862712039, decimal=10)
        npt.assert_almost_equal(values[2][1], -0.056609097000281938, decimal=10)


if __name__ == '__main__':
    pytest.main()

__author__ = 'lynevdv'


from lenstronomy.LensModel.Profiles.elliptical_density_slice import ElliSLICE

import numpy as np
import pytest
import numpy.testing as npt

class TestElliSLICE(object):
    """
    tests tthe elliptical slice lens model
    """
    def setup(self):
        self.ElliSLICE = ElliSLICE()

    def test_function(self):
        x = 0.5
        y = 0.1
        a = 2.
        b = 1.
        psi = 30*np.pi/180.
        sigma_0 = 5.
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values, 4.532482297, decimal=4)
        x = np.array([0])
        y = np.array([0])
        values = self.ElliSLICE.function(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(values[0], 4.054651081,decimal=5)

        x = np.array([np.sqrt(3), np.sqrt(3)+0.000000001, np.sqrt(3)-0.000000001])
        y = np.array([1, 1.000000001, 0.999999999])
        values = self.ElliSLICE.function(x, y,  a, b, psi, sigma_0)
        npt.assert_almost_equal(values[0], values[1], decimal=5)
        npt.assert_almost_equal(values[1], values[2], decimal=5)

    def test_derivatives(self):
        x = 0.5
        y = 0.1
        a = 2.
        b = 1.
        psi = 30 * np.pi / 180.
        sigma_0 = 5.
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x, 1.938995765, decimal=6)
        npt.assert_almost_equal(f_y, -0.13835403, decimal=6)

        x = np.array([0.5])
        y = np.array([0.1])
        f_x, f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x,  1.938995765, decimal=6)
        npt.assert_almost_equal(f_y,  -0.13835403, decimal=6)

        x = np.array([np.sqrt(3), np.sqrt(3) + 0.000000001, np.sqrt(3) - 0.000000001])
        y = np.array([1, 1.000000001, 0.999999999])
        f_x,f_y = self.ElliSLICE.derivatives(x, y, a, b, psi, sigma_0)
        npt.assert_almost_equal(f_x[0], f_x[1], decimal=5)
        npt.assert_almost_equal(f_y[1], f_y[2], decimal=5)


    def test_hessian(self):
        x = 1
        y = 2
        m = 4
        a_m = 0.05
        phi_m = 25 * np.pi / 180.
        f_xx, f_yy, f_xy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx, -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy, -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy, 0.008021169, decimal=6)
        x = np.array([1])
        y = np.array([2])
        f_xx, f_yy,f_xy = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(f_xx[0], -0.016042338, decimal=6)
        npt.assert_almost_equal(f_yy[0], -0.004010584, decimal=6)
        npt.assert_almost_equal(f_xy[0], 0.008021169, decimal=6)
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.Multipole.hessian(x, y, m, a_m, phi_m)
        npt.assert_almost_equal(values[0][0], -0.016042338, decimal=6)
        npt.assert_almost_equal(values[1][0], -0.004010584, decimal=6)
        npt.assert_almost_equal(values[2][0], 0.008021169, decimal=6)
        npt.assert_almost_equal(values[0][1], 0.001417956, decimal=6)
        npt.assert_almost_equal(values[1][1], 0.012761602, decimal=6)
        npt.assert_almost_equal(values[2][1], -0.004253867, decimal=6)


if __name__ == '__main__':
   pytest.main()

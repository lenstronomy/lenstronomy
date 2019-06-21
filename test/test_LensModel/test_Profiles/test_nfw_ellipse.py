__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.Profiles.nfw_ellipse import NFW_ELLIPSE
import lenstronomy.Util.param_util as param_util

import numpy as np
import numpy.testing as npt
import pytest

class TestNFWELLIPSE(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.nfw = NFW()
        self.nfw_e = NFW_ELLIPSE()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        alpha_Rs = 1.
        q = 1.
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.nfw.function(x, y, Rs, alpha_Rs)
        values_e = self.nfw_e.function(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(values[0], values_e[0], decimal=5)
        x = np.array([0])
        y = np.array([0])

        q = .8
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.nfw_e.function(x, y, Rs, alpha_Rs,e1, e2)
        npt.assert_almost_equal(values[0], 0, decimal=4)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.nfw_e.function(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(values[0], 1.8827504143588476, decimal=5)
        npt.assert_almost_equal(values[1], 2.6436373117941852, decimal=5)
        npt.assert_almost_equal(values[2], 3.394127018818891, decimal=5)

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        alpha_Rs = 1.
        q = 1.
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_x, f_y = self.nfw.derivatives(x, y, Rs, alpha_Rs)
        f_x_e, f_y_e = self.nfw_e.derivatives(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(f_x[0], f_x_e[0], decimal=5)
        npt.assert_almost_equal(f_y[0], f_y_e[0], decimal=5)
        x = np.array([0])
        y = np.array([0])
        alpha_Rs = 0
        f_x, f_y = self.nfw_e.derivatives(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(f_x[0], 0, decimal=5)
        npt.assert_almost_equal(f_y[0], 0, decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        alpha_Rs = 1.
        q = .8
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.nfw_e.derivatives(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(values[0][0], 0.32458737284934414, decimal=5)
        npt.assert_almost_equal(values[1][0], 0.9737621185480323, decimal=5)
        npt.assert_almost_equal(values[0][1], 0.76249351329615234, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.38124675664807617, decimal=5)

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        alpha_Rs = 1.
        q = 1.
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        f_xx, f_yy,f_xy = self.nfw.hessian(x, y, Rs, alpha_Rs)
        f_xx_e, f_yy_e, f_xy_e = self.nfw_e.hessian(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(f_xx[0], f_xx_e[0], decimal=5)
        npt.assert_almost_equal(f_yy[0], f_yy_e[0], decimal=5)
        npt.assert_almost_equal(f_xy[0], f_xy_e[0], decimal=5)

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        q = .8
        phi_G = 0
        e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
        values = self.nfw_e.hessian(x, y, Rs, alpha_Rs, e1, e2)
        npt.assert_almost_equal(values[0][0], 0.26998576668768592, decimal=5)
        npt.assert_almost_equal(values[1][0], -0.0045328224507201753, decimal=5)
        npt.assert_almost_equal(values[2][0], -0.16380454531672584, decimal=5)
        npt.assert_almost_equal(values[0][1], -0.014833136829928151, decimal=5)
        npt.assert_almost_equal(values[1][1], 0.31399726446723619, decimal=5)
        npt.assert_almost_equal(values[2][1], -0.13449884961325154, decimal=5)


if __name__ == '__main__':
    pytest.main()

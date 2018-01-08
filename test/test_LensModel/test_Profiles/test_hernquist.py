__author__ = 'sibirrer'

import pytest
import numpy as np
import numpy.testing as npt

from lenstronomy.LensModel.Profiles.hernquist import Hernquist
from lenstronomy.LensModel.Profiles.hernquist_ellipse import Hernquist_Ellipse


class TestHernquist(object):

    def setup(self):
        self.profile = Hernquist()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function(x, y, sigma0, Rs)
        assert values[0] == 0.66514613455415028
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function( x, y, sigma0, Rs)
        npt.assert_almost_equal(values[0], 0, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.profile.function( x, y, sigma0, Rs)
        assert values[0] == 0.66514613455415028
        assert values[1] == 0.87449395673649566
        assert values[2] == 1.0549139073851708

    def test_derivatives(self):
        x = 1
        y = 2
        Rs = 1.
        sigma0 = 0.5
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs)
        assert f_x == 0.11160641027573866
        assert f_y == 0.22321282055147731
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs)
        assert f_x[0] == 0
        assert f_y[0] == 0


    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        f_xx, f_yy,f_xy = self.profile.hessian(x, y, sigma0, Rs)
        assert f_xx[0] == 0.07790120765543973
        assert f_yy[0] == -0.023212946451134361
        assert f_xy[0] == -0.067409341317214988


class TestHernquistEllipse(object):

    def setup(self):
        self.profile = Hernquist_Ellipse()

    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        values = self.profile.function(x, y, sigma0, Rs,q, phi_G)
        assert values[0] == 0.6451374041763912
        x = np.array([0])
        y = np.array([0])
        Rs = 1.
        sigma0 = 0.5
        values = self.profile.function( x, y, sigma0, Rs, q, phi_G)
        npt.assert_almost_equal(values[0], 0, decimal=6)

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.profile.function( x, y, sigma0, Rs, q, phi_G)
        assert values[0] == 0.60477384241056542
        assert values[1] == 0.80854098526603968
        assert values[2] == 0.98780932325084092

    def test_derivatives(self):
        x = 1
        y = 2
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs, q, phi_G)
        assert f_x == 0.065446024625706908
        assert f_y == 0.24132860718623173
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.profile.derivatives( x, y, sigma0, Rs, q, phi_G)
        assert f_x[0] == 0
        assert f_y[0] == 0


    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        Rs = 1.
        sigma0 = 0.5
        q, phi_G = 0.8, 0.5
        f_xx, f_yy,f_xy = self.profile.hessian(x, y, sigma0, Rs, q, phi_G)
        assert f_xx[0] == 0.093409372903252574
        assert f_yy[0] == -0.028538260199439947
        assert f_xy[0] == -0.06298435105411837


if __name__ == '__main__':
    pytest.main()

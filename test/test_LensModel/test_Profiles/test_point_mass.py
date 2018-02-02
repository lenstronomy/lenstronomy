__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.point_mass import PointMass

import numpy as np
import pytest

class TestSIS(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.pointmass = PointMass()


    def test_function(self):
        x = np.array([0])
        y = np.array([1])
        theta_E = 1.
        values = self.pointmass.function(x, y, theta_E)
        assert values[0] == 0
        x = np.array([0])
        y = np.array([0])
        values = self.pointmass.function(x, y, theta_E)
        assert values[0] < 0

        x = np.array([1,3,4])
        y = np.array([0,1,1])
        values = self.pointmass.function( x, y, theta_E)
        assert values[0] == 0
        assert values[1] == 1.151292546497023
        assert values[2] == 1.4166066720281081

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = 1.
        f_x, f_y = self.pointmass.derivatives(x, y, theta_E)
        assert f_x[0] == 1
        assert f_y[0] == 0
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.pointmass.derivatives(x, y, theta_E)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([0,1,1])
        values = self.pointmass.derivatives(x, y, theta_E)
        assert values[0][0] == 1
        assert values[1][0] == 0
        assert values[0][1] == 0.29999999999999999
        assert values[1][1] == 0.099999999999999992

    def test_hessian(self):
        x = np.array([1])
        y = np.array([0])
        theta_E = 1.
        f_xx, f_yy,f_xy = self.pointmass.hessian(x, y, theta_E)
        assert f_xx[0] == -1
        assert f_yy[0] == 1
        assert f_xy[0] == -0
        x = np.array([1,3,4])
        y = np.array([0,1,1])
        values = self.pointmass.hessian(x, y, theta_E)
        assert values[0][0] == -1
        assert values[1][0] == 1
        assert values[2][0] == -0
        assert values[0][1] == -0.080000000000000002
        assert values[1][1] == 0.080000000000000002
        assert values[2][1] == -0.059999999999999998


if __name__ == '__main__':
    pytest.main()
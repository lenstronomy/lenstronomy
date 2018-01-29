__author__ = 'sibirrer'

from lenstronomy.LensModel.Profiles.sis_truncate import SIS_truncate

import numpy as np
import pytest

class TestSIS_truncate(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SIS = SIS_truncate()


    def test_function(self):
        x = 1
        y = 0
        phi_E = 1.
        r_trunc = 2
        values = self.SIS.function(x, y, phi_E, r_trunc)
        assert values == 1
        x = np.array([0])
        y = np.array([0])
        phi_E = 1.
        values = self.SIS.function(x, y, phi_E, r_trunc)
        assert values[0] == 0

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.SIS.function(x, y, phi_E, r_trunc)
        assert values[0] == 2.2221359549995796
        assert values[1] == 2.8245553203367586
        assert values[2] == 3

    def test_derivatives(self):
        x = 1
        y = 2
        phi_E = 1.
        r_trunc = 2
        f_x, f_y = self.SIS.derivatives(x, y, phi_E, r_trunc)
        assert f_x == 0.39442719099991586
        assert f_y == 0.78885438199983171
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.SIS.derivatives(x, y, phi_E, r_trunc)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.SIS.derivatives(x, y, phi_E, r_trunc)
        assert values[0][0] == 0.39442719099991586
        assert values[1][0] == 0.78885438199983171
        assert values[0][1] == 0.39736659610102748
        assert values[1][1] == 0.13245553203367583

    def test_hessian(self):
        x = 1
        y = 0
        phi_E = 1.
        r_trunc = 2
        f_xx, f_yy,f_xy = self.SIS.hessian(x, y, phi_E, r_trunc)
        assert f_xx == 0
        assert f_yy == 1
        assert f_xy == 0
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.SIS.hessian(x, y, phi_E, r_trunc)
        assert values[0][0] == 0.21554175279993265
        assert values[1][0] == -0.3211145618000168
        assert values[2][0] == -0.3577708763999663
        assert values[0][1] == -0.43675444679663239
        assert values[1][1] == 0.06920997883030823
        assert values[2][1] == -0.18973665961010272


if __name__ == '__main__':
    pytest.main()
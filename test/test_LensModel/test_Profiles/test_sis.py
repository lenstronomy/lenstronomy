__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.sis import SIS

import numpy as np
import pytest

class TestSIS(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.SIS = SIS()


    def test_function(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        values = self.SIS.function(x, y, phi_E)
        assert values[0] == 2.2360679774997898
        x = np.array([0])
        y = np.array([0])
        phi_E = 1.
        values = self.SIS.function( x, y, phi_E)
        assert values[0] == 0

        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.SIS.function( x, y, phi_E)
        assert values[0] == 2.2360679774997898
        assert values[1] == 3.1622776601683795
        assert values[2] == 4.1231056256176606

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        f_x, f_y = self.SIS.derivatives( x, y, phi_E)
        assert f_x[0] == 0.44721359549995793
        assert f_y[0] == 0.89442719099991586
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.SIS.derivatives( x, y, phi_E)
        assert f_x[0] == 0
        assert f_y[0] == 0

        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.SIS.derivatives(x, y, phi_E)
        assert values[0][0] == 0.44721359549995793
        assert values[1][0] == 0.89442719099991586
        assert values[0][1] == 0.94868329805051377
        assert values[1][1] == 0.31622776601683794

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        f_xx, f_yy,f_xy = self.SIS.hessian( x, y, phi_E)
        assert f_xx[0] == 0.35777087639996635
        assert f_yy[0] == 0.089442719099991588
        assert f_xy[0] == -0.17888543819998318
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.SIS.hessian( x, y, phi_E)
        assert values[0][0] == 0.35777087639996635
        assert values[1][0] == 0.089442719099991588
        assert values[2][0] == -0.17888543819998318
        assert values[0][1] == 0.031622776601683791
        assert values[1][1] == 0.28460498941515411
        assert values[2][1] == -0.094868329805051374


if __name__ == '__main__':
    pytest.main()
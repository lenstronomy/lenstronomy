__author__ = 'sibirrer'


from lenstronomy.LensModel.Profiles.no_lens import NoLens

import numpy as np
import pytest

class TestSIS(object):
    """
    tests the Gaussian methods
    """
    def setup(self):
        self.noLens = NoLens()


    def test_function(self):
        x = np.array([1])
        y = np.array([2])

        values = self.noLens.function(x, y, **{})
        assert values[0] == 0


        x = np.array([2,3,4])
        y = np.array([1,1,1])
        values = self.noLens.function(x, y)
        assert values[0] == 0
        assert values[1] == 0
        assert values[2] == 0

    def test_derivatives(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        f_x, f_y = self.noLens.derivatives(x, y, **{})
        assert f_x[0] == 0
        assert f_y[0] == 0
        x = np.array([0])
        y = np.array([0])
        f_x, f_y = self.noLens.derivatives(x, y)
        assert f_x[0] == 0
        assert f_y[0] == 0

    def test_hessian(self):
        x = np.array([1])
        y = np.array([2])
        phi_E = 1.
        f_xx, f_yy, f_xy = self.noLens.hessian( x, y, **{})
        assert f_xx[0] == 0
        assert f_yy[0] == 0
        assert f_xy[0] == 0
        x = np.array([1,3,4])
        y = np.array([2,1,1])
        values = self.noLens.hessian(x, y)
        assert values[0][0] == 0
        assert values[1][0] == 0
        assert values[2][0] == 0
        assert values[0][1] == 0
        assert values[1][1] == 0
        assert values[2][1] == 0


if __name__ == '__main__':
    pytest.main()
